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
    assert cfg.noise.audit_modes == ("ideal", "shots", "aer_noise")
    assert int(cfg.noise.shots) == 2048
    assert int(cfg.noise.oracle_repeats) == 4
    assert str(cfg.noise.oracle_aggregate) == "mean"
    assert cfg.noise.mitigation_config == {"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None}
    assert cfg.noise.symmetry_mitigation_config == {
        "mode": "off",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }
    assert bool(cfg.noise.include_final_audit) is False
    assert bool(cfg.noise.include_full_circuit_audit) is False
    assert bool(cfg.fixed_lean_replay.enabled) is False
    assert bool(cfg.fixed_scaffold_replay.enabled) is False
    assert str(cfg.source.mode) == "fresh_stage"
    assert str(cfg.staged.artifacts.tag).startswith("hh_staged_noise_")
    assert Path(cfg.staged.artifacts.output_json).name == f"{cfg.staged.artifacts.tag}.json"
    assert Path(cfg.staged.artifacts.replay_output_json).name == f"{cfg.staged.artifacts.tag}_replay.json"


def test_explicit_noise_tag_is_preserved() -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(["--L", "2", "--skip-pdf", "--tag", "custom_noise_tag"])
    )

    assert str(cfg.staged.artifacts.tag) == "custom_noise_tag"
    assert Path(cfg.staged.artifacts.output_json).name == "custom_noise_tag.json"


def test_resolve_oracle_v1_defaults_controller_noise_mode_to_backend_scheduled_with_fake_backend() -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--checkpoint-controller-mode",
                "oracle_v1",
                "--use-fake-backend",
            ]
        )
    )

    assert str(cfg.staged.realtime_checkpoint.mode) == "oracle_v1"
    assert str(cfg.noise.controller_noise_mode) == "backend_scheduled"


def test_resolve_oracle_v1_requires_explicit_controller_noise_mode_when_multiple_modes() -> None:
    with pytest.raises(ValueError, match="checkpoint-controller-noise-mode"):
        noise_wf.resolve_staged_hh_noise_config(
            parse_args(
                [
                    "--L",
                    "2",
                    "--skip-pdf",
                    "--checkpoint-controller-mode",
                    "oracle_v1",
                ]
            )
        )


def test_resolve_oracle_v1_accepts_runtime_controller_noise_mode() -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--checkpoint-controller-mode",
                "oracle_v1",
                "--checkpoint-controller-noise-mode",
                "runtime",
                "--backend-name",
                "ibm_marrakesh",
            ]
        )
    )

    assert str(cfg.noise.controller_noise_mode) == "runtime"
    assert str(cfg.noise.backend_name) == "ibm_marrakesh"
    assert bool(cfg.noise.use_fake_backend) is False


def test_resolve_oracle_v1_rejects_backend_scheduled_readout_with_active_symmetry() -> None:
    with pytest.raises(ValueError, match="active symmetry mitigation"):
        noise_wf.resolve_staged_hh_noise_config(
            parse_args(
                [
                    "--L",
                    "2",
                    "--skip-pdf",
                    "--checkpoint-controller-mode",
                    "oracle_v1",
                    "--use-fake-backend",
                    "--mitigation",
                    "readout",
                    "--symmetry-mitigation-mode",
                    "postselect_diag_v1",
                ]
            )
        )


def test_resolve_imported_artifact_rejects_checkpoint_controller_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "lean.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_full_circuit_import_json", lambda: (source_json, True))

    with pytest.raises(ValueError, match="fresh-stage noisy runs"):
        noise_wf.resolve_staged_hh_noise_config(
            parse_args(
                [
                    "--L",
                    "2",
                    "--skip-pdf",
                    "--include-full-circuit-audit",
                    "--checkpoint-controller-mode",
                    "oracle_v1",
                ]
            )
        )


def test_resolve_imported_full_circuit_defaults_to_lean_subject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "lean.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_full_circuit_import_json", lambda: (source_json, True))

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(["--L", "2", "--skip-pdf", "--include-full-circuit-audit"])
    )

    assert str(cfg.source.mode) == "imported_artifact"
    assert cfg.source.resolved_json == source_json
    assert bool(cfg.source.default_subject) is True
    assert cfg.noise.audit_modes == ("ideal", "backend_scheduled")
    assert cfg.noise.backend_name == "FakeGuadalupeV2"
    assert bool(cfg.noise.use_fake_backend) is True


def test_resolve_imported_full_circuit_accepts_local_gate_twirling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "lean.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_full_circuit_import_json", lambda: (source_json, True))

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-full-circuit-audit",
                "--local-gate-twirling",
            ]
        )
    )

    assert str(cfg.source.mode) == "imported_artifact"
    assert cfg.noise.mitigation_config["local_gate_twirling"] is True


def test_resolve_fixed_lean_noisy_replay_defaults_to_imported_subject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "lean.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_full_circuit_import_json", lambda: (source_json, True))

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-lean-noisy-replay",
                "--final-method",
                "SPSA",
            ]
        )
    )

    assert str(cfg.source.mode) == "imported_artifact"
    assert cfg.source.resolved_json == source_json
    assert bool(cfg.fixed_lean_replay.enabled) is True
    assert int(cfg.fixed_lean_replay.reps) == 1
    assert str(cfg.fixed_lean_replay.method) == "SPSA"
    assert str(cfg.fixed_lean_replay.noise_mode) == "backend_scheduled"
    assert cfg.fixed_lean_replay.mitigation_config == {
        "mode": "readout",
        "zne_scales": [],
        "dd_sequence": None,
        "local_readout_strategy": "mthree",
    }


def test_resolve_fixed_lean_noisy_replay_accepts_powell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "lean.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_full_circuit_import_json", lambda: (source_json, True))

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-lean-noisy-replay",
                "--final-method",
                "Powell",
            ]
        )
    )

    assert bool(cfg.fixed_lean_replay.enabled) is True
    assert str(cfg.fixed_lean_replay.method) == "Powell"
    assert str(cfg.fixed_lean_replay.noise_mode) == "backend_scheduled"


def test_resolve_fixed_lean_noise_attribution_defaults_to_imported_subject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "lean.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_full_circuit_import_json", lambda: (source_json, True))

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-lean-noise-attribution",
            ]
        )
    )

    assert str(cfg.source.mode) == "imported_artifact"
    assert cfg.source.resolved_json == source_json
    assert bool(cfg.fixed_lean_attribution.enabled) is True
    assert cfg.fixed_lean_attribution.slices == (
        "readout_only",
        "gate_stateprep_only",
        "full",
    )
    assert str(cfg.fixed_lean_attribution.noise_mode) == "backend_scheduled"
    assert cfg.fixed_lean_attribution.mitigation_config["mode"] == "none"
    assert cfg.noise.backend_name == "FakeGuadalupeV2"
    assert bool(cfg.noise.use_fake_backend) is True


def test_resolve_fixed_lean_compile_control_scout_defaults_to_imported_subject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "lean.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_full_circuit_import_json", lambda: (source_json, True))

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-lean-compile-control-scout",
                "--backend-name",
                "FakeHeron",
            ]
        )
    )

    assert str(cfg.source.mode) == "imported_artifact"
    assert cfg.source.resolved_json == source_json
    assert bool(cfg.fixed_lean_compile_control_scout.enabled) is True
    assert cfg.fixed_lean_compile_control_scout.baseline_transpile_optimization_level == 1
    assert cfg.fixed_lean_compile_control_scout.baseline_seed_transpiler == 7
    assert cfg.fixed_lean_compile_control_scout.scout_transpile_optimization_levels == (1, 2)
    assert cfg.fixed_lean_compile_control_scout.scout_seed_transpilers == (0, 1, 2, 3, 4)
    assert cfg.fixed_lean_compile_control_scout.mitigation_config == {
        "mode": "readout",
        "zne_scales": [],
        "dd_sequence": None,
        "local_readout_strategy": "mthree",
    }
    assert cfg.fixed_lean_compile_control_scout.symmetry_mitigation_config == {
        "mode": "off",
        "num_sites": None,
        "ordering": "blocked",
        "sector_n_up": None,
        "sector_n_dn": None,
    }
    assert cfg.noise.backend_name == "FakeHeron"
    assert bool(cfg.noise.use_fake_backend) is True


def test_resolve_fixed_scaffold_noisy_replay_defaults_to_marrakesh_subject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "fixed_scaffold.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_fixed_scaffold_runtime_import_json", lambda: (source_json, True))

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-scaffold-noisy-replay",
                "--use-fake-backend",
                "--final-method",
                "SPSA",
            ]
        )
    )

    assert str(cfg.source.mode) == "imported_artifact"
    assert cfg.source.resolved_json == source_json
    assert cfg.source.default_subject_kind == "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    assert bool(cfg.fixed_scaffold_replay.enabled) is True
    assert str(cfg.fixed_scaffold_replay.subject_kind) == "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    assert str(cfg.noise.backend_name) == "FakeMarrakesh"
    assert bool(cfg.noise.use_fake_backend) is True
    assert str(cfg.fixed_scaffold_replay.noise_mode) == "backend_scheduled"
    assert cfg.fixed_scaffold_replay.mitigation_config["mode"] == "readout"
    assert cfg.fixed_scaffold_replay.local_dd_probe_sequence is None


def test_resolve_fixed_scaffold_noisy_replay_dd_sequence_becomes_local_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "fixed_scaffold.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_fixed_scaffold_runtime_import_json", lambda: (source_json, True))

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-scaffold-noisy-replay",
                "--use-fake-backend",
                "--dd-sequence",
                "XpXm",
            ]
        )
    )

    assert cfg.fixed_scaffold_replay.local_dd_probe_sequence == "XPXM"
    assert cfg.fixed_scaffold_replay.mitigation_config["dd_sequence"] is None


def test_resolve_fixed_scaffold_noise_attribution_defaults_to_nighthawk_subject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "fixed_scaffold.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_fixed_scaffold_import_json", lambda: (source_json, True))

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-scaffold-noise-attribution",
            ]
        )
    )

    assert str(cfg.source.mode) == "imported_artifact"
    assert cfg.source.resolved_json == source_json
    assert cfg.source.default_subject_kind == "hh_nighthawk_gate_pruned_7term_v1"
    assert bool(cfg.fixed_scaffold_attribution.enabled) is True
    assert str(cfg.fixed_scaffold_attribution.subject_kind) == "hh_nighthawk_gate_pruned_7term_v1"
    assert cfg.fixed_scaffold_attribution.mitigation_config["mode"] == "none"
    assert cfg.noise.backend_name == "FakeNighthawk"
    assert bool(cfg.noise.use_fake_backend) is True


def test_resolve_fixed_scaffold_compile_control_scout_defaults_to_nighthawk_subject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "fixed_scaffold.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_fixed_scaffold_import_json", lambda: (source_json, True))

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-scaffold-compile-control-scout",
                "--backend-name",
                "FakeNighthawk",
            ]
        )
    )

    assert str(cfg.source.mode) == "imported_artifact"
    assert cfg.source.resolved_json == source_json
    assert cfg.source.default_subject_kind == "hh_nighthawk_gate_pruned_7term_v1"
    assert bool(cfg.fixed_scaffold_compile_control_scout.enabled) is True
    assert str(cfg.fixed_scaffold_compile_control_scout.subject_kind) == "hh_nighthawk_gate_pruned_7term_v1"
    assert cfg.fixed_scaffold_compile_control_scout.baseline_transpile_optimization_level == 2
    assert cfg.fixed_scaffold_compile_control_scout.baseline_seed_transpiler == 7
    assert cfg.fixed_scaffold_compile_control_scout.scout_transpile_optimization_levels == (1, 2)
    assert cfg.fixed_scaffold_compile_control_scout.scout_seed_transpilers == (0, 1, 2, 3, 4)
    assert cfg.fixed_scaffold_compile_control_scout.mitigation_config == {
        "mode": "readout",
        "zne_scales": [],
        "dd_sequence": None,
        "local_readout_strategy": "mthree",
    }
    assert cfg.fixed_scaffold_compile_control_scout.symmetry_mitigation_config == {
        "mode": "off",
        "num_sites": None,
        "ordering": "blocked",
        "sector_n_up": None,
        "sector_n_dn": None,
    }
    assert cfg.noise.backend_name == "FakeNighthawk"
    assert bool(cfg.noise.use_fake_backend) is True


def test_resolve_fixed_scaffold_compile_control_scout_requires_backend_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "fixed_scaffold.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_fixed_scaffold_import_json", lambda: (source_json, True))

    with pytest.raises(ValueError, match="fixed scaffold compile-control scout requires an explicit --backend-name"):
        noise_wf.resolve_staged_hh_noise_config(
            parse_args(
                [
                    "--L",
                    "2",
                    "--skip-pdf",
                    "--include-fixed-scaffold-compile-control-scout",
                ]
            )
        )


def test_resolve_fixed_scaffold_runtime_energy_only_defaults_to_marrakesh_subject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "runtime_fixed_scaffold.json"
    source_json.write_text(
        "{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        noise_wf, "_default_fixed_scaffold_runtime_import_json", lambda: (source_json, True)
    )

    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-scaffold-runtime-energy-only-baseline",
                "--backend-name",
                "ibm_marrakesh",
            ]
        )
    )

    assert str(cfg.source.mode) == "imported_artifact"
    assert cfg.source.resolved_json == source_json
    assert cfg.source.default_subject_kind == "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    assert bool(cfg.fixed_scaffold_runtime_energy_only.enabled) is True
    assert (
        str(cfg.fixed_scaffold_runtime_energy_only.subject_kind)
        == "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    )
    assert str(cfg.noise.backend_name) == "ibm_marrakesh"
    assert bool(cfg.noise.use_fake_backend) is False
    assert str(cfg.fixed_scaffold_runtime_energy_only.noise_mode) == "runtime"
    assert cfg.fixed_scaffold_runtime_energy_only.runtime_profile_config["name"] == "main_twirled_readout_v1"
    assert cfg.fixed_scaffold_runtime_energy_only.runtime_session_config == {
        "mode": "require_session"
    }
    assert cfg.fixed_scaffold_runtime_energy_only.transpile_optimization_level == 1
    assert cfg.fixed_scaffold_runtime_energy_only.seed_transpiler == 0
    assert cfg.fixed_scaffold_runtime_energy_only.include_dd_probe is False
    assert cfg.fixed_scaffold_runtime_energy_only.include_final_zne_audit is False


def test_resolve_fixed_scaffold_noisy_replay_requires_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "runtime_fixed_scaffold.json"
    source_json.write_text(
        "{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        noise_wf, "_default_fixed_scaffold_runtime_import_json", lambda: (source_json, True)
    )

    with pytest.raises(ValueError, match="local-only on the fake-backend path"):
        noise_wf.resolve_staged_hh_noise_config(
            parse_args(
                [
                    "--L",
                    "2",
                    "--skip-pdf",
                    "--include-fixed-scaffold-noisy-replay",
                    "--backend-name",
                    "ibm_marrakesh",
                    "--final-method",
                    "SPSA",
                ]
            )
        )


def test_resolve_fixed_lean_and_fixed_scaffold_flags_conflict() -> None:
    with pytest.raises(ValueError, match="cannot be enabled together"):
        noise_wf.resolve_staged_hh_noise_config(
            parse_args(
                [
                    "--L",
                    "2",
                    "--skip-pdf",
                    "--include-fixed-lean-noisy-replay",
                    "--include-fixed-scaffold-noisy-replay",
                ]
            )
        )


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


def test_run_staged_hh_noise_fresh_stage_wires_oracle_controller_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--checkpoint-controller-mode",
                "oracle_v1",
                "--use-fake-backend",
                "--output-json",
                str(tmp_path / "noise_controller.json"),
                "--output-pdf",
                str(tmp_path / "noise_controller.pdf"),
            ]
        )
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(noise_wf.base_wf, "run_stage_pipeline", lambda staged_cfg: _stage_result())
    monkeypatch.setattr(noise_wf.base_wf, "run_noiseless_profiles", lambda stage_result_arg, staged_cfg: {"profiles": {}})
    monkeypatch.setattr(noise_wf, "run_noisy_profiles", lambda stage_result_arg, cfg_arg: ({"profiles": {}}, {"profiles": {}}, {"rows": []}))
    monkeypatch.setattr(
        noise_wf,
        "run_adaptive_realtime_checkpoint_profile_noisy",
        lambda stage_result_arg, cfg_arg: {"mode": "oracle_v1", "status": "completed", "summary": {"append_count": 1, "stay_count": 2, "decision_noise_mode": "backend_scheduled"}},
    )
    def _fake_assemble_payload(**kwargs):
        captured["kwargs"] = kwargs
        return {
            "pipeline": "hh_staged_noiseless",
            "workflow_contract": {},
            "settings": {},
            "artifacts": {"workflow": {"output_json": str(cfg.staged.artifacts.output_json), "output_pdf": str(cfg.staged.artifacts.output_pdf)}},
            "stage_pipeline": {},
            "dynamics_noiseless": kwargs["dynamics_noiseless"],
            "adaptive_realtime_checkpoint": kwargs["adaptive_realtime_checkpoint"],
        }

    monkeypatch.setattr(noise_wf.base_wf, "assemble_payload", _fake_assemble_payload)
    monkeypatch.setattr(noise_wf.base_wf, "_compute_comparisons", lambda payload: {})
    monkeypatch.setattr(noise_wf.noise_report, "_compute_comparisons", lambda payload: {})
    monkeypatch.setattr(noise_wf.base_wf, "_write_json", lambda path, payload: None)
    monkeypatch.setattr(noise_wf, "write_staged_hh_noise_pdf", lambda payload, cfg_arg, run_command: None)

    payload = noise_wf.run_staged_hh_noise(cfg, run_command="python pipelines/hardcoded/hh_staged_noise.py --checkpoint-controller-mode oracle_v1")

    assert captured["kwargs"]["adaptive_realtime_checkpoint"]["mode"] == "oracle_v1"
    assert payload["adaptive_realtime_checkpoint"]["summary"]["append_count"] == 1


def test_run_staged_hh_noise_import_mode_skips_stage_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "lean.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-full-circuit-audit",
                "--fixed-final-state-json",
                str(source_json),
                "--output-json",
                str(tmp_path / "import_noise.json"),
                "--output-pdf",
                str(tmp_path / "import_noise.pdf"),
            ]
        )
    )

    monkeypatch.setattr(
        noise_wf.base_wf,
        "run_stage_pipeline",
        lambda staged_cfg: (_ for _ in ()).throw(AssertionError("stage pipeline should not run in import mode")),
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_prepared_state_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "prepared", "mode": mode},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_full_circuit_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "full", "mode": mode},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_lean_noisy_replay_mode",
        lambda source_cfg, noise_cfg, fixed_cfg: {"success": True, "route": "fixed_lean_scaffold_noisy_replay"},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_lean_noise_attribution_mode",
        lambda source_cfg, noise_cfg, attribution_cfg: {"success": True, "route": "fixed_lean_noise_attribution"},
    )
    writes: dict[str, object] = {}
    monkeypatch.setattr(noise_wf.base_wf, "_write_json", lambda path, payload: writes.update(path=Path(path), payload=payload))
    monkeypatch.setattr(noise_wf, "write_staged_hh_noise_pdf", lambda payload, cfg_arg, run_command: writes.setdefault("pdf", True))

    payload = noise_wf.run_staged_hh_noise(cfg, run_command="python pipelines/hardcoded/hh_staged_noise.py --include-full-circuit-audit")

    assert payload["workflow_contract"]["noise_extension"] == "imported_adapt_circuit_audit"
    assert payload["import_source"]["mode"] == "imported_artifact"
    assert payload["imported_prepared_state_audit"]["modes"]["ideal"]["success"] is True
    assert payload["full_circuit_import_audit"]["modes"]["backend_scheduled"]["success"] is True
    assert payload["fixed_lean_scaffold_noisy_replay"] == {}
    assert payload["fixed_lean_noise_attribution"] == {}
    assert writes["path"] == cfg.staged.artifacts.output_json
    assert bool(writes["pdf"]) is True


def test_run_staged_hh_noise_import_mode_runs_fixed_lean_noisy_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "lean.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-lean-noisy-replay",
                "--fixed-final-state-json",
                str(source_json),
                "--final-method",
                "SPSA",
                "--output-json",
                str(tmp_path / "import_noise.json"),
                "--output-pdf",
                str(tmp_path / "import_noise.pdf"),
            ]
        )
    )

    monkeypatch.setattr(
        noise_wf.base_wf,
        "run_stage_pipeline",
        lambda staged_cfg: (_ for _ in ()).throw(AssertionError("stage pipeline should not run in import mode")),
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_prepared_state_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "prepared", "mode": mode},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_full_circuit_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "full", "mode": mode},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_lean_noisy_replay_mode",
        lambda source_cfg, noise_cfg, fixed_cfg: {
            "success": True,
            "route": "fixed_lean_scaffold_noisy_replay",
            "energies": {"best_noisy_minus_ideal": 0.123},
        },
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_lean_noise_attribution_mode",
        lambda source_cfg, noise_cfg, attribution_cfg: {
            "success": True,
            "route": "fixed_lean_noise_attribution",
            "slices": {
                "readout_only": {"success": True, "delta_mean": 0.2},
                "gate_stateprep_only": {"success": True, "delta_mean": 1.1},
                "full": {"success": True, "delta_mean": 1.3},
            },
        },
    )
    writes: dict[str, object] = {}
    monkeypatch.setattr(noise_wf.base_wf, "_write_json", lambda path, payload: writes.update(path=Path(path), payload=payload))
    monkeypatch.setattr(noise_wf, "write_staged_hh_noise_pdf", lambda payload, cfg_arg, run_command: writes.setdefault("pdf", True))

    payload = noise_wf.run_staged_hh_noise(cfg, run_command="python pipelines/hardcoded/hh_staged_noise.py --include-fixed-lean-noisy-replay")

    assert payload["workflow_contract"]["imported_routes"]["fixed_lean_scaffold_noisy_replay"] is True
    assert payload["workflow_contract"]["imported_routes"]["fixed_lean_noise_attribution"] is False
    assert payload["settings"]["fixed_lean_noisy_replay"]["enabled"] is True
    assert payload["settings"]["fixed_lean_noise_attribution"]["enabled"] is False
    assert payload["fixed_lean_scaffold_noisy_replay"]["success"] is True
    assert payload["summary"]["fixed_lean_scaffold_noisy_replay_completed"] == 1
    assert writes["path"] == cfg.staged.artifacts.output_json


def test_run_staged_hh_noise_import_mode_runs_fixed_scaffold_noisy_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "fixed_scaffold.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-scaffold-noisy-replay",
                "--use-fake-backend",
                "--fixed-final-state-json",
                str(source_json),
                "--final-method",
                "SPSA",
                "--output-json",
                str(tmp_path / "import_noise.json"),
                "--output-pdf",
                str(tmp_path / "import_noise.pdf"),
            ]
        )
    )

    monkeypatch.setattr(
        noise_wf.base_wf,
        "run_stage_pipeline",
        lambda staged_cfg: (_ for _ in ()).throw(AssertionError("stage pipeline should not run in import mode")),
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_prepared_state_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "prepared", "mode": mode},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_full_circuit_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "full", "mode": mode},
    )
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noise_attribution_mode", lambda source_cfg, noise_cfg, attribution_cfg: {})
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_scaffold_noisy_replay_mode",
        lambda source_cfg, noise_cfg, fixed_cfg: {
            "success": True,
            "route": "fixed_scaffold_noisy_replay",
            "subject_kind": "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1",
            "term_order_id": "source_order_pruned",
            "theta_source": "imported_theta_runtime",
            "execution_mode": "backend_scheduled",
            "local_mitigation_label": "readout_only",
            "energies": {"best_noisy_minus_ideal": 0.456},
        },
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_scaffold_noise_attribution_mode",
        lambda source_cfg, noise_cfg, attribution_cfg: {},
    )
    writes: dict[str, object] = {}
    monkeypatch.setattr(noise_wf.base_wf, "_write_json", lambda path, payload: writes.update(path=Path(path), payload=payload))
    monkeypatch.setattr(noise_wf, "write_staged_hh_noise_pdf", lambda payload, cfg_arg, run_command: writes.setdefault("pdf", True))

    payload = noise_wf.run_staged_hh_noise(cfg, run_command="python pipelines/hardcoded/hh_staged_noise.py --include-fixed-scaffold-noisy-replay")

    assert payload["workflow_contract"]["imported_routes"]["fixed_scaffold_noisy_replay"] is True
    assert payload["workflow_contract"]["imported_routes"]["fixed_scaffold_noise_attribution"] is False
    assert payload["settings"]["fixed_scaffold_noisy_replay"]["enabled"] is True
    assert payload["settings"]["fixed_scaffold_noise_attribution"]["enabled"] is False
    assert payload["fixed_scaffold_noisy_replay"]["success"] is True
    assert payload["summary"]["fixed_scaffold_noisy_replay_completed"] == 1
    assert writes["path"] == cfg.staged.artifacts.output_json


def test_run_staged_hh_noise_import_mode_runs_fixed_scaffold_runtime_energy_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "runtime_fixed_scaffold.json"
    source_json.write_text(
        "{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}",
        encoding="utf-8",
    )
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-scaffold-runtime-energy-only-baseline",
                "--fixed-final-state-json",
                str(source_json),
                "--backend-name",
                "ibm_marrakesh",
                "--output-json",
                str(tmp_path / "runtime_energy_only.json"),
                "--output-pdf",
                str(tmp_path / "runtime_energy_only.pdf"),
            ]
        )
    )

    monkeypatch.setattr(
        noise_wf.base_wf,
        "run_stage_pipeline",
        lambda staged_cfg: (_ for _ in ()).throw(
            AssertionError("stage pipeline should not run in import mode")
        ),
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_prepared_state_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "prepared", "mode": mode},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_full_circuit_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "full", "mode": mode},
    )
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noise_attribution_mode", lambda source_cfg, noise_cfg, attribution_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_compile_control_scout_mode", lambda source_cfg, noise_cfg, scout_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_compile_control_scout_mode", lambda source_cfg, noise_cfg, scout_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_noise_attribution_mode", lambda source_cfg, noise_cfg, attribution_cfg: {})
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_scaffold_runtime_energy_only_mode",
        lambda source_cfg, noise_cfg, runtime_cfg: {
            "success": True,
            "route": "fixed_scaffold_runtime_energy_only",
            "subject_kind": "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1",
            "energy_audits": {"main": {"success": True, "evaluation": {"delta_mean": 0.234}}},
        },
    )

    writes: dict[str, object] = {}

    def _fake_write_json(path: Path, payload: dict[str, object]) -> None:
        writes[str(Path(path).name)] = payload
        writes.setdefault("paths", []).append(Path(path))

    monkeypatch.setattr(noise_wf.base_wf, "_write_json", _fake_write_json)
    monkeypatch.setattr(
        noise_wf,
        "write_staged_hh_noise_pdf",
        lambda payload, cfg_arg, run_command: writes.setdefault("pdf", True),
    )

    payload = noise_wf.run_staged_hh_noise(
        cfg,
        run_command="python pipelines/hardcoded/hh_staged_noise.py --include-fixed-scaffold-runtime-energy-only-baseline --backend-name ibm_marrakesh",
    )

    assert payload["workflow_contract"]["imported_routes"]["fixed_scaffold_runtime_energy_only"] is True
    assert payload["settings"]["fixed_scaffold_runtime_energy_only"]["enabled"] is True
    assert payload["fixed_scaffold_runtime_energy_only"]["success"] is True
    assert payload["summary"]["fixed_scaffold_runtime_energy_only_completed"] == 1
    assert payload["summary"]["fixed_scaffold_runtime_energy_only_main_delta_mean"] == pytest.approx(0.234)
    assert payload["artifacts"]["fixed_scaffold_runtime_energy_only_json"] is not None
    sidecar_name = Path(payload["artifacts"]["fixed_scaffold_runtime_energy_only_json"]).name
    assert sidecar_name in writes
    sidecar_payload = writes[sidecar_name]
    assert sidecar_payload["pipeline"] == "hh_fixed_scaffold_energy_only_runtime_eval_v1"
    assert sidecar_payload["backend_name"] == "ibm_marrakesh"
    assert bool(writes["pdf"]) is True


def test_run_staged_hh_noise_import_mode_runs_fixed_lean_noise_attribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "lean.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-lean-noise-attribution",
                "--fixed-final-state-json",
                str(source_json),
                "--output-json",
                str(tmp_path / "import_attr.json"),
                "--output-pdf",
                str(tmp_path / "import_attr.pdf"),
            ]
        )
    )

    monkeypatch.setattr(
        noise_wf.base_wf,
        "run_stage_pipeline",
        lambda staged_cfg: (_ for _ in ()).throw(AssertionError("stage pipeline should not run in import mode")),
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_prepared_state_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "prepared", "mode": mode},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_full_circuit_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "full", "mode": mode},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_lean_noisy_replay_mode",
        lambda source_cfg, noise_cfg, fixed_cfg: {},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_lean_noise_attribution_mode",
        lambda source_cfg, noise_cfg, attribution_cfg: {
            "success": True,
            "route": "fixed_lean_noise_attribution",
            "slices": {
                "readout_only": {"success": True, "delta_mean": 0.2},
                "gate_stateprep_only": {"success": True, "delta_mean": 1.1},
                "full": {"success": True, "delta_mean": 1.3},
            },
        },
    )
    writes: dict[str, object] = {}
    monkeypatch.setattr(noise_wf.base_wf, "_write_json", lambda path, payload: writes.update(path=Path(path), payload=payload))
    monkeypatch.setattr(noise_wf, "write_staged_hh_noise_pdf", lambda payload, cfg_arg, run_command: writes.setdefault("pdf", True))

    payload = noise_wf.run_staged_hh_noise(cfg, run_command="python pipelines/hardcoded/hh_staged_noise.py --include-fixed-lean-noise-attribution")

    assert payload["workflow_contract"]["imported_routes"]["fixed_lean_noise_attribution"] is True
    assert payload["settings"]["fixed_lean_noise_attribution"]["enabled"] is True
    assert payload["fixed_lean_noise_attribution"]["success"] is True
    assert payload["summary"]["fixed_lean_noise_attribution_completed"] == 1
    assert payload["summary"]["fixed_lean_noise_attribution_slices_completed"] == 3
    assert writes["path"] == cfg.staged.artifacts.output_json


def test_run_staged_hh_noise_import_mode_runs_fixed_scaffold_noise_attribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "fixed_scaffold.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-scaffold-noise-attribution",
                "--fixed-final-state-json",
                str(source_json),
                "--output-json",
                str(tmp_path / "import_attr.json"),
                "--output-pdf",
                str(tmp_path / "import_attr.pdf"),
            ]
        )
    )

    monkeypatch.setattr(
        noise_wf.base_wf,
        "run_stage_pipeline",
        lambda staged_cfg: (_ for _ in ()).throw(AssertionError("stage pipeline should not run in import mode")),
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_prepared_state_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "prepared", "mode": mode},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_full_circuit_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "full", "mode": mode},
    )
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noise_attribution_mode", lambda source_cfg, noise_cfg, attribution_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_scaffold_noise_attribution_mode",
        lambda source_cfg, noise_cfg, attribution_cfg: {
            "success": True,
            "route": "fixed_scaffold_noise_attribution",
            "subject_kind": "hh_nighthawk_gate_pruned_7term_v1",
            "term_order_id": "source_order",
            "slices": {
                "readout_only": {"success": True, "delta_mean": 0.2},
                "gate_stateprep_only": {"success": True, "delta_mean": 1.1},
                "full": {"success": True, "delta_mean": 1.3},
            },
        },
    )
    writes: dict[str, object] = {}
    monkeypatch.setattr(noise_wf.base_wf, "_write_json", lambda path, payload: writes.update(path=Path(path), payload=payload))
    monkeypatch.setattr(noise_wf, "write_staged_hh_noise_pdf", lambda payload, cfg_arg, run_command: writes.setdefault("pdf", True))

    payload = noise_wf.run_staged_hh_noise(cfg, run_command="python pipelines/hardcoded/hh_staged_noise.py --include-fixed-scaffold-noise-attribution")

    assert payload["workflow_contract"]["imported_routes"]["fixed_scaffold_noise_attribution"] is True
    assert payload["settings"]["fixed_scaffold_noise_attribution"]["enabled"] is True
    assert payload["fixed_scaffold_noise_attribution"]["success"] is True
    assert payload["summary"]["fixed_scaffold_noise_attribution_completed"] == 1
    assert payload["summary"]["fixed_scaffold_noise_attribution_slices_completed"] == 3
    assert writes["path"] == cfg.staged.artifacts.output_json


def test_run_staged_hh_noise_import_mode_runs_fixed_lean_compile_control_scout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "lean.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-lean-compile-control-scout",
                "--fixed-final-state-json",
                str(source_json),
                "--backend-name",
                "FakeHeron",
                "--output-json",
                str(tmp_path / "import_scout.json"),
                "--output-pdf",
                str(tmp_path / "import_scout.pdf"),
            ]
        )
    )

    monkeypatch.setattr(
        noise_wf.base_wf,
        "run_stage_pipeline",
        lambda staged_cfg: (_ for _ in ()).throw(AssertionError("stage pipeline should not run in import mode")),
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_prepared_state_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "prepared", "mode": mode},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_full_circuit_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "full", "mode": mode},
    )
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noise_attribution_mode", lambda source_cfg, noise_cfg, attribution_cfg: {})
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_lean_compile_control_scout_mode",
        lambda source_cfg, noise_cfg, scout_cfg: {
            "success": True,
            "route": "fixed_lean_compile_control_scout",
            "candidate_counts": {"total": 10, "successful": 10, "failed": 0},
            "best_candidate": {
                "label": "opt2_seed1",
                "delta_mean": 0.42,
                "compiled_two_qubit_count": 18,
                "compiled_depth": 61,
            },
        },
    )
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_noise_attribution_mode", lambda source_cfg, noise_cfg, attribution_cfg: {})
    writes: dict[str, object] = {}
    monkeypatch.setattr(noise_wf.base_wf, "_write_json", lambda path, payload: writes.update(path=Path(path), payload=payload))
    monkeypatch.setattr(noise_wf, "write_staged_hh_noise_pdf", lambda payload, cfg_arg, run_command: writes.setdefault("pdf", True))

    payload = noise_wf.run_staged_hh_noise(cfg, run_command="python pipelines/hardcoded/hh_staged_noise.py --include-fixed-lean-compile-control-scout --backend-name FakeHeron")

    assert payload["workflow_contract"]["imported_routes"]["fixed_lean_compile_control_scout"] is True
    assert payload["settings"]["fixed_lean_compile_control_scout"]["enabled"] is True
    assert payload["fixed_lean_compile_control_scout"]["success"] is True
    assert payload["summary"]["fixed_lean_compile_control_scout_completed"] == 1
    assert payload["summary"]["fixed_lean_compile_control_scout_candidates_successful"] == 10
    assert payload["summary"]["fixed_lean_compile_control_scout_best_two_qubit_count"] == pytest.approx(18.0)
    assert writes["path"] == cfg.staged.artifacts.output_json


def test_run_staged_hh_noise_import_mode_runs_fixed_scaffold_compile_control_scout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "fixed_scaffold.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-scaffold-compile-control-scout",
                "--fixed-final-state-json",
                str(source_json),
                "--backend-name",
                "FakeNighthawk",
                "--output-json",
                str(tmp_path / "import_scout.json"),
                "--output-pdf",
                str(tmp_path / "import_scout.pdf"),
            ]
        )
    )

    monkeypatch.setattr(
        noise_wf.base_wf,
        "run_stage_pipeline",
        lambda staged_cfg: (_ for _ in ()).throw(AssertionError("stage pipeline should not run in import mode")),
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_prepared_state_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "prepared", "mode": mode},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_full_circuit_audit_mode",
        lambda source_cfg, noise_cfg, mode: {"success": True, "kind": "full", "mode": mode},
    )
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noise_attribution_mode", lambda source_cfg, noise_cfg, attribution_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_compile_control_scout_mode", lambda source_cfg, noise_cfg, scout_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_noise_attribution_mode", lambda source_cfg, noise_cfg, attribution_cfg: {})
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_scaffold_compile_control_scout_mode",
        lambda source_cfg, noise_cfg, scout_cfg: {
            "success": True,
            "route": "fixed_scaffold_compile_control_scout",
            "subject_kind": "hh_nighthawk_gate_pruned_7term_v1",
            "term_order_id": "source_order",
            "candidate_counts": {"total": 10, "successful": 8, "failed": 2},
            "best_candidate": {
                "label": "opt2_seed7",
                "delta_mean": 0.31,
                "compiled_two_qubit_count": 19,
                "compiled_depth": 57,
            },
        },
    )
    writes: dict[str, object] = {}
    monkeypatch.setattr(noise_wf.base_wf, "_write_json", lambda path, payload: writes.update(path=Path(path), payload=payload))
    monkeypatch.setattr(noise_wf, "write_staged_hh_noise_pdf", lambda payload, cfg_arg, run_command: writes.setdefault("pdf", True))

    payload = noise_wf.run_staged_hh_noise(
        cfg,
        run_command="python pipelines/hardcoded/hh_staged_noise.py --include-fixed-scaffold-compile-control-scout --backend-name FakeNighthawk",
    )

    assert payload["workflow_contract"]["imported_routes"]["fixed_scaffold_compile_control_scout"] is True
    assert payload["settings"]["fixed_scaffold_compile_control_scout"]["enabled"] is True
    assert payload["fixed_scaffold_compile_control_scout"]["success"] is True
    assert payload["summary"]["fixed_scaffold_compile_control_scout_completed"] == 1
    assert payload["summary"]["fixed_scaffold_compile_control_scout_candidates_successful"] == 8
    assert payload["summary"]["fixed_scaffold_compile_control_scout_best_two_qubit_count"] == pytest.approx(19.0)
    assert writes["path"] == cfg.staged.artifacts.output_json


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


def test_noise_cli_main_print_contract_import_mode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake_cfg = SimpleNamespace(staged=SimpleNamespace(artifacts=SimpleNamespace(skip_pdf=True)))
    fake_payload = {
        "artifacts": {
            "workflow": {"output_json": "artifacts/json/hh_staged_noise.json", "output_pdf": "artifacts/pdf/hh_staged_noise.pdf"},
            "import_source_json": "artifacts/json/adapt_hh_L2.json",
            "intermediate": {
                "adapt_handoff_json": "artifacts/json/hh_staged_noise_adapt_handoff.json",
                "replay_output_json": "artifacts/json/hh_staged_noise_replay.json",
            },
        },
        "import_source": {"mode": "imported_artifact"},
    }

    monkeypatch.setattr(noise_cli, "resolve_staged_hh_noise_config", lambda args: fake_cfg)
    monkeypatch.setattr(noise_cli, "run_staged_hh_noise", lambda cfg: fake_payload)

    noise_cli.main(["--skip-pdf"])
    lines = capsys.readouterr().out.strip().splitlines()

    assert lines == [
        "workflow_json=artifacts/json/hh_staged_noise.json",
        "import_source_json=artifacts/json/adapt_hh_L2.json",
        "adapt_handoff_json=artifacts/json/hh_staged_noise_adapt_handoff.json",
    ]
