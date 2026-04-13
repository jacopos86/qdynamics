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

import pipelines.hardcoded.hh_staged_noise as noise_cli
import pipelines.hardcoded.hh_staged_noise_workflow as noise_wf
import pipelines.hardcoded.hh_staged_workflow as base_wf
from pipelines.hardcoded.hh_staged_noise import parse_args
from pipelines.hardcoded.hh_realtime_checkpoint_types import ScaffoldAcceptanceResult


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

    assert cfg.noise.methods == ("suzuki2",)
    assert cfg.noise.modes == ("ideal", "shots", "aer_noise")
    assert cfg.noise.audit_modes == ("ideal", "shots", "aer_noise")
    assert cfg.noise.controller_noise_mode is None
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
    assert str(cfg.staged.adapt.continuation_mode) == "phase3_v1"
    assert str(cfg.staged.replay.continuation_mode) == "phase3_v1"
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


def test_resolve_oracle_v1_accepts_controller_tuning_and_tier_overrides() -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--checkpoint-controller-mode",
                "oracle_v1",
                "--checkpoint-controller-noise-mode",
                "backend_scheduled",
                "--use-fake-backend",
                "--backend-name",
                "FakeMarrakesh",
                "--checkpoint-controller-shortlist-size",
                "1",
                "--checkpoint-controller-oracle-selection-policy",
                "measured_topk_oracle_energy",
                "--checkpoint-controller-candidate-step-scales",
                "0.25,0.5,1.0",
                "--checkpoint-controller-exact-forecast-guardrail-mode",
                "dual_metric_v1",
                "--checkpoint-controller-exact-forecast-fidelity-loss-tol",
                "0.01",
                "--checkpoint-controller-exact-forecast-abs-energy-error-increase-tol",
                "0.02",
                "--checkpoint-controller-max-probe-positions",
                "1",
                "--checkpoint-controller-motion-calm-shortlist-scale",
                "0.25",
                "--checkpoint-controller-motion-kink-shortlist-bonus",
                "0",
                "--checkpoint-controller-scout-shots",
                "8",
                "--checkpoint-controller-confirm-shots",
                "16",
                "--checkpoint-controller-commit-shots",
                "16",
                "--checkpoint-controller-scout-repeats",
                "1",
                "--checkpoint-controller-confirm-repeats",
                "1",
                "--checkpoint-controller-commit-repeats",
                "1",
                "--checkpoint-controller-timeout-s",
                "300",
                "--checkpoint-controller-progress-every-s",
                "2.5",
            ]
        )
    )

    assert int(cfg.staged.realtime_checkpoint.shortlist_size) == 1
    assert str(cfg.staged.realtime_checkpoint.oracle_selection_policy) == "measured_topk_oracle_energy"
    assert tuple(cfg.staged.realtime_checkpoint.candidate_step_scales) == pytest.approx((0.25, 0.5, 1.0))
    assert str(cfg.staged.realtime_checkpoint.exact_forecast_guardrail_mode) == "dual_metric_v1"
    assert float(cfg.staged.realtime_checkpoint.exact_forecast_fidelity_loss_tol) == pytest.approx(0.01)
    assert float(cfg.staged.realtime_checkpoint.exact_forecast_abs_energy_error_increase_tol) == pytest.approx(0.02)
    assert int(cfg.staged.realtime_checkpoint.max_probe_positions) == 1
    assert float(cfg.staged.realtime_checkpoint.motion_calm_shortlist_scale) == pytest.approx(0.25)
    assert int(cfg.staged.realtime_checkpoint.motion_kink_shortlist_bonus) == 0
    assert str(cfg.staged.realtime_checkpoint.confirm_score_mode) == "compressed_whitened_v1"
    assert float(cfg.staged.realtime_checkpoint.confirm_compress_fraction) == pytest.approx(0.5)
    assert str(cfg.staged.realtime_checkpoint.prune_mode) == "off"
    assert float(cfg.staged.realtime_checkpoint.prune_loss_threshold) == pytest.approx(0.01)
    assert int(cfg.staged.realtime_checkpoint.tiers[0].oracle_shots) == 8
    assert int(cfg.staged.realtime_checkpoint.tiers[1].oracle_shots) == 16
    assert int(cfg.staged.realtime_checkpoint.tiers[2].oracle_shots) == 16
    assert int(cfg.noise.controller_timeout_s) == 300
    assert float(cfg.noise.controller_progress_every_s) == pytest.approx(2.5)


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
    monkeypatch.setattr(noise_wf, "_default_fixed_scaffold_import_json", lambda: (source_json, True))

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


def test_resolve_fixed_scaffold_noisy_replay_symmetry_active_keeps_none_mitigation(
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
                "--symmetry-mitigation-mode",
                "projector_renorm_v1",
            ]
        )
    )

    assert cfg.fixed_scaffold_replay.mitigation_config["mode"] == "none"
    assert cfg.fixed_scaffold_replay.symmetry_mitigation_config["mode"] == "projector_renorm_v1"


def test_resolve_fixed_scaffold_noisy_replay_inherits_explicit_symmetry_and_zne(
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
                "--mitigation",
                "zne",
                "--zne-scales",
                "1.0,2.0,3.0",
                "--symmetry-mitigation-mode",
                "projector_renorm_v1",
            ]
        )
    )

    assert cfg.fixed_scaffold_replay.mitigation_config["mode"] == "zne"
    assert cfg.fixed_scaffold_replay.mitigation_config["zne_scales"] == [1.0, 2.0, 3.0]
    assert cfg.fixed_scaffold_replay.symmetry_mitigation_config["mode"] == "projector_renorm_v1"


def test_resolve_fixed_scaffold_noisy_replay_rejects_in_loop_dd_mitigation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "fixed_scaffold.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_fixed_scaffold_runtime_import_json", lambda: (source_json, True))

    with pytest.raises(ValueError, match="saved-theta local probe only"):
        noise_wf.resolve_staged_hh_noise_config(
            parse_args(
                [
                    "--L",
                    "2",
                    "--skip-pdf",
                    "--include-fixed-scaffold-noisy-replay",
                    "--use-fake-backend",
                    "--mitigation",
                    "dd",
                    "--dd-sequence",
                    "XpXm",
                ]
            )
        )


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


def test_resolve_fixed_scaffold_compile_control_scout_defaults_to_marrakesh_subject(
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
                "--include-fixed-scaffold-compile-control-scout",
                "--backend-name",
                "FakeMarrakesh",
            ]
        )
    )

    assert str(cfg.source.mode) == "imported_artifact"
    assert cfg.source.resolved_json == source_json
    assert cfg.source.default_subject_kind == "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    assert bool(cfg.fixed_scaffold_compile_control_scout.enabled) is True
    assert str(cfg.fixed_scaffold_compile_control_scout.subject_kind) == "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    assert cfg.fixed_scaffold_compile_control_scout.baseline_transpile_optimization_level == 1
    assert cfg.fixed_scaffold_compile_control_scout.baseline_seed_transpiler == 0
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
    assert cfg.noise.backend_name == "FakeMarrakesh"
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


def test_resolve_fixed_scaffold_saved_theta_mitigation_matrix_defaults_to_marrakesh_subject(
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
                "--include-fixed-scaffold-saved-theta-mitigation-matrix",
            ]
        )
    )

    assert str(cfg.source.mode) == "imported_artifact"
    assert cfg.source.resolved_json == source_json
    assert cfg.source.default_subject_kind == "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    assert bool(cfg.fixed_scaffold_saved_theta_mitigation_matrix.enabled) is True
    assert (
        str(cfg.fixed_scaffold_saved_theta_mitigation_matrix.subject_kind)
        == "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    )
    assert str(cfg.fixed_scaffold_saved_theta_mitigation_matrix.noise_mode) == "backend_scheduled"
    assert cfg.fixed_scaffold_saved_theta_mitigation_matrix.compile_presets == (
        {"label": "opt1_seed4", "transpile_optimization_level": 1, "seed_transpiler": 4},
        {"label": "opt2_seed0", "transpile_optimization_level": 2, "seed_transpiler": 0},
    )
    assert cfg.fixed_scaffold_saved_theta_mitigation_matrix.zne_scales == (1.0, 3.0, 5.0)
    assert cfg.fixed_scaffold_saved_theta_mitigation_matrix.suppression_labels == (
        "readout_plus_gate_twirling",
        "readout_plus_local_dd",
        "readout_plus_gate_twirling_plus_local_dd",
    )
    assert cfg.fixed_scaffold_saved_theta_mitigation_matrix.selected_cells == ()
    assert cfg.fixed_scaffold_saved_theta_mitigation_matrix.mitigation_config_base == {
        "mode": "readout",
        "zne_scales": [],
        "dd_sequence": None,
        "local_readout_strategy": "mthree",
    }
    assert cfg.fixed_scaffold_saved_theta_mitigation_matrix.symmetry_mitigation_config == {
        "mode": "off",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }
    assert cfg.noise.backend_name == "FakeMarrakesh"
    assert bool(cfg.noise.use_fake_backend) is True


def test_resolve_fixed_scaffold_saved_theta_mitigation_matrix_allows_custom_presets_and_cells(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "weak_fixed_scaffold.json"
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
                "--include-fixed-scaffold-saved-theta-mitigation-matrix",
                "--fixed-scaffold-matrix-compile-presets",
                "opt2_seed5:2:5",
                "--fixed-scaffold-matrix-selected-cells",
                (
                    "opt2_seed5__zne_on__twirl_dd,"
                    "opt2_seed5__zne_on__twirl,"
                    "opt2_seed5__zne_on__dd,"
                    "opt2_seed5__zne_off__twirl_dd"
                ),
                "--fixed-scaffold-matrix-base-mitigation-mode",
                "none",
                "--symmetry-mitigation-mode",
                "projector_renorm_v1",
                "--sector-n-up",
                "1",
                "--sector-n-dn",
                "1",
                "--fixed-final-state-json",
                str(source_json),
            ]
        )
    )

    assert cfg.fixed_scaffold_saved_theta_mitigation_matrix.compile_presets == (
        {"label": "opt2_seed5", "transpile_optimization_level": 2, "seed_transpiler": 5},
    )
    assert cfg.fixed_scaffold_saved_theta_mitigation_matrix.selected_cells == (
        "opt2_seed5__zne_on__twirl_dd",
        "opt2_seed5__zne_on__twirl",
        "opt2_seed5__zne_on__dd",
        "opt2_seed5__zne_off__twirl_dd",
    )
    assert cfg.fixed_scaffold_saved_theta_mitigation_matrix.mitigation_config_base == {
        "mode": "none",
        "zne_scales": [],
        "dd_sequence": None,
        "local_readout_strategy": None,
    }
    assert cfg.fixed_scaffold_saved_theta_mitigation_matrix.symmetry_mitigation_config == {
        "mode": "projector_renorm_v1",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }


def test_resolve_fixed_scaffold_saved_theta_mitigation_matrix_rejects_runtime_only_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "fixed_scaffold.json"
    source_json.write_text("{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}", encoding="utf-8")
    monkeypatch.setattr(noise_wf, "_default_fixed_scaffold_runtime_import_json", lambda: (source_json, True))

    with pytest.raises(ValueError, match="does not support the legacy Runtime final ZNE audit flag"):
        noise_wf.resolve_staged_hh_noise_config(
            parse_args(
                [
                    "--L",
                    "2",
                    "--skip-pdf",
                    "--include-fixed-scaffold-saved-theta-mitigation-matrix",
                    "--include-fixed-scaffold-runtime-final-zne-audit",
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


def test_resolve_fixed_scaffold_runtime_raw_baseline_defaults_to_marrakesh_subject(
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
                "--include-fixed-scaffold-runtime-raw-baseline",
                "--backend-name",
                "ibm_marrakesh",
            ]
        )
    )

    assert str(cfg.source.mode) == "imported_artifact"
    assert cfg.source.resolved_json == source_json
    assert cfg.source.default_subject_kind == "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    assert bool(cfg.fixed_scaffold_runtime_raw_baseline.enabled) is True
    assert (
        str(cfg.fixed_scaffold_runtime_raw_baseline.subject_kind)
        == "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    )
    assert str(cfg.noise.backend_name) == "ibm_marrakesh"
    assert bool(cfg.noise.use_fake_backend) is False
    assert str(cfg.fixed_scaffold_runtime_raw_baseline.noise_mode) == "runtime"
    assert cfg.fixed_scaffold_runtime_raw_baseline.mitigation_config["mode"] == "none"
    assert cfg.fixed_scaffold_runtime_raw_baseline.runtime_profile_config["name"] == "legacy_runtime_v0"
    assert cfg.fixed_scaffold_runtime_raw_baseline.runtime_session_config == {
        "mode": "require_session"
    }
    assert cfg.fixed_scaffold_runtime_raw_baseline.transpile_optimization_level == 1
    assert cfg.fixed_scaffold_runtime_raw_baseline.seed_transpiler == 0
    assert cfg.fixed_scaffold_runtime_raw_baseline.raw_transport == "auto"
    assert bool(cfg.fixed_scaffold_runtime_raw_baseline.raw_store_memory) is False
    assert cfg.fixed_scaffold_runtime_raw_baseline.raw_artifact_path is None


def test_resolve_fixed_scaffold_runtime_raw_baseline_local_fake_backend_preserves_requested_diagonal_postprocessing(
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
                "--include-fixed-scaffold-runtime-raw-baseline",
                "--use-fake-backend",
                "--mitigation",
                "readout",
                "--symmetry-mitigation-mode",
                "projector_renorm_v1",
            ]
        )
    )

    assert str(cfg.source.mode) == "imported_artifact"
    assert cfg.source.resolved_json == source_json
    assert str(cfg.noise.backend_name) == "FakeMarrakesh"
    assert bool(cfg.noise.use_fake_backend) is True
    assert str(cfg.fixed_scaffold_runtime_raw_baseline.noise_mode) == "backend_scheduled"
    assert cfg.fixed_scaffold_runtime_raw_baseline.mitigation_config["mode"] == "readout"
    assert cfg.fixed_scaffold_runtime_raw_baseline.mitigation_config["local_readout_strategy"] == "mthree"
    assert cfg.fixed_scaffold_runtime_raw_baseline.symmetry_mitigation_config["mode"] == "projector_renorm_v1"
    assert cfg.fixed_scaffold_runtime_raw_baseline.raw_transport == "auto"
    assert cfg.fixed_scaffold_runtime_raw_baseline.runtime_profile_config["name"] == "legacy_runtime_v0"


def test_resolve_fixed_scaffold_runtime_raw_baseline_accepts_explicit_sampler_profile_and_verify_only(
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
                "--include-fixed-scaffold-runtime-raw-baseline",
                "--backend-name",
                "ibm_marrakesh",
                "--fixed-scaffold-runtime-raw-profile",
                "raw_sampler_twirled_v1",
                "--symmetry-mitigation-mode",
                "verify_only",
            ]
        )
    )

    assert bool(cfg.fixed_scaffold_runtime_raw_baseline.enabled) is True
    assert cfg.fixed_scaffold_runtime_raw_baseline.runtime_profile_config["name"] == "raw_sampler_twirled_v1"
    assert cfg.fixed_scaffold_runtime_raw_baseline.symmetry_mitigation_config["mode"] == "verify_only"


def test_resolve_imported_artifact_rejects_phase3_oracle_gradient_knobs(
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

    with pytest.raises(ValueError, match="phase3 oracle-gradient knobs"):
        noise_wf.resolve_staged_hh_noise_config(
            parse_args(
                [
                    "--L",
                    "2",
                    "--skip-pdf",
                    "--adapt-continuation-mode",
                    "phase3_v1",
                    "--include-fixed-scaffold-runtime-raw-baseline",
                    "--backend-name",
                    "ibm_marrakesh",
                    "--phase3-oracle-gradient-mode",
                    "runtime",
                    "--phase3-oracle-backend-name",
                    "ibm_marrakesh",
                ]
            )
        )


def test_build_noise_summary_preserves_fixed_scaffold_saved_theta_mitigation_matrix_fields() -> None:
    payload = {
        "stage_pipeline": {},
        "dynamics_noisy": {"profiles": {}},
        "fixed_scaffold_saved_theta_mitigation_matrix": {
            "success": True,
            "cell_counts": {
                "total": 12,
                "completed": 12,
                "successful": 12,
                "failed": 0,
            },
            "best_cell": {
                "label": "opt2_seed0__zne_on__twirl_dd",
                "delta_mean": 0.04,
                "compiled_two_qubit_count": 14,
                "compiled_depth": 38,
                "compile_preset_label": "opt2_seed0",
                "suppression_stack": "readout_plus_gate_twirling_plus_local_dd",
                "mitigation_config": {"mode": "readout"},
                "zne_enabled": True,
            },
        },
    }

    summary = noise_wf._build_noise_summary(payload)

    assert summary["fixed_scaffold_saved_theta_mitigation_matrix_completed"] == 1
    assert summary["fixed_scaffold_saved_theta_mitigation_matrix_total"] == 1
    assert summary["fixed_scaffold_saved_theta_mitigation_matrix_cells_total"] == 12
    assert summary["fixed_scaffold_saved_theta_mitigation_matrix_cells_completed"] == 12
    assert summary["fixed_scaffold_saved_theta_mitigation_matrix_cells_failed"] == 0
    assert summary["fixed_scaffold_saved_theta_mitigation_matrix_best_label"] == "opt2_seed0__zne_on__twirl_dd"
    assert summary["fixed_scaffold_saved_theta_mitigation_matrix_best_delta_mean"] == pytest.approx(0.04)
    assert summary["fixed_scaffold_saved_theta_mitigation_matrix_best_compile_preset"] == "opt2_seed0"
    assert summary["fixed_scaffold_saved_theta_mitigation_matrix_best_suppression_stack"] == (
        "readout_plus_gate_twirling_plus_local_dd"
    )
    assert summary["fixed_scaffold_saved_theta_mitigation_matrix_best_zne_enabled"] is True
    assert summary["fixed_scaffold_saved_theta_mitigation_matrix_best_mitigation_mode"] == "readout"


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


def test_resolve_fixed_scaffold_runtime_pairing_allows_both_routes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "runtime_pairing.json"
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
                "--include-fixed-scaffold-runtime-raw-baseline",
                "--fixed-final-state-json",
                str(source_json),
                "--backend-name",
                "ibm_marrakesh",
                "--fixed-scaffold-runtime-raw-profile",
                "raw_sampler_twirled_v1",
                "--symmetry-mitigation-mode",
                "verify_only",
            ]
        )
    )

    assert bool(cfg.fixed_scaffold_runtime_energy_only.enabled) is True
    assert bool(cfg.fixed_scaffold_runtime_raw_baseline.enabled) is True
    assert str(cfg.fixed_scaffold_runtime_energy_only.noise_mode) == "runtime"
    assert str(cfg.fixed_scaffold_runtime_raw_baseline.noise_mode) == "runtime"


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
        lambda dynamics_noisy: [{"profile": "static", "method": "suzuki2", "mode": "ideal"}],
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
                "suzuki2",
            ]
        )
    )
    stage_result = _stage_result()
    dynamics_noisy, noisy_final_audit, dynamics_benchmarks = noise_wf.run_noisy_profiles(stage_result, cfg)

    assert set(dynamics_noisy["profiles"].keys()) == {"static", "drive"}
    assert set(noisy_final_audit["profiles"].keys()) == {"static", "drive"}
    assert dynamics_benchmarks["rows"] == [{"profile": "static", "method": "suzuki2", "mode": "ideal"}]
    assert len(mode_calls) == 4
    assert len(audit_calls) == 4
    assert all(np.allclose(rec["kwargs"]["psi_seed"], stage_result.psi_final) for rec in mode_calls)
    assert all(np.allclose(rec["kwargs"]["psi_seed"], stage_result.psi_final) for rec in audit_calls)
    assert {str(rec["kwargs"]["method"]) for rec in mode_calls} == {"suzuki2"}
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
                "suzuki2",
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
                            "suzuki2": {
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
                            "suzuki2": {
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
            {"rows": [{"profile": "static", "method": "suzuki2", "mode": "ideal"}]},
        ),
    )
    monkeypatch.setattr(
        noise_wf,
        "run_adaptive_realtime_checkpoint_profile_noisy",
        lambda stage_result_arg, cfg_arg: None,
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
    assert payload["settings"]["noise"]["methods"] == ["suzuki2"]
    assert payload["dynamics_benchmarks"]["rows"] == [{"profile": "static", "method": "suzuki2", "mode": "ideal"}]
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


def test_prepare_adaptive_realtime_checkpoint_inputs_allows_drive_for_oracle_v1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--enable-drive",
                "--checkpoint-controller-mode",
                "oracle_v1",
                "--checkpoint-controller-noise-mode",
                "backend_scheduled",
                "--use-fake-backend",
                "--backend-name",
                "FakeMarrakesh",
            ]
        )
    )
    stage_result = _stage_result()
    stage_result = base_wf.StageExecutionResult(
        **{
            **stage_result.__dict__,
            "replay_payload": {
                **dict(stage_result.replay_payload),
                "best_state": {"best_theta": [0.1]},
            },
        }
    )

    fake_context = SimpleNamespace(payload_in={"accepted": True})
    fake_acceptance = SimpleNamespace(accepted=True, reason="ok", source_kind="pytest")

    monkeypatch.setattr(base_wf.replay_mod, "build_replay_scaffold_context", lambda replay_cfg, h_poly: fake_context)
    monkeypatch.setattr(base_wf, "validate_scaffold_acceptance", lambda payload_in: fake_acceptance)

    prepared = base_wf._prepare_adaptive_realtime_checkpoint_inputs(stage_result, cfg.staged)

    assert prepared is not None
    replay_context, acceptance, best_theta = prepared
    assert replay_context is fake_context
    assert acceptance is fake_acceptance
    assert list(best_theta) == [0.1]


def test_run_adaptive_realtime_checkpoint_profile_noisy_forwards_drive_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--output-json",
                str(tmp_path / "noise.json"),
                "--output-pdf",
                str(tmp_path / "noise.pdf"),
                "--enable-drive",
                "--drive-A",
                "0.6",
                "--drive-omega",
                "1.0",
                "--drive-tbar",
                "1.0",
                "--drive-phi",
                "0.0",
                "--drive-pattern",
                "staggered",
                "--drive-time-sampling",
                "midpoint",
                "--drive-t0",
                "0.0",
                "--exact-steps-multiplier",
                "1",
                "--checkpoint-controller-mode",
                "oracle_v1",
                "--checkpoint-controller-noise-mode",
                "backend_scheduled",
                "--use-fake-backend",
                "--backend-name",
                "FakeMarrakesh",
                "--checkpoint-controller-timeout-s",
                "123",
                "--checkpoint-controller-progress-every-s",
                "2.0",
            ]
        )
    )
    stage_result = _stage_result()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        noise_wf,
        "_prepare_checkpoint_controller_inputs",
        lambda stage_result_arg, staged_cfg_arg: (
            SimpleNamespace(payload_in={}),
            ScaffoldAcceptanceResult(
                accepted=True,
                reason="ok",
                structure_locked=True,
                source_kind="pytest",
            ),
            [0.1],
        ),
    )

    def _fake_isolated(**kwargs):
        captured.update(kwargs)
        return {"mode": "oracle_v1", "status": "completed", "summary": {"append_count": 0, "stay_count": 0}}

    monkeypatch.setattr(noise_wf, "_run_adaptive_realtime_checkpoint_profile_noisy_isolated", _fake_isolated)

    payload = noise_wf.run_adaptive_realtime_checkpoint_profile_noisy(stage_result, cfg)

    assert payload is not None
    controller_kwargs = captured["controller_kwargs"]
    assert controller_kwargs["drive_config"] is not None
    drive_cfg = controller_kwargs["drive_config"]
    assert bool(drive_cfg.enabled) is True
    assert int(drive_cfg.n_sites) == 2
    assert str(drive_cfg.ordering) == "blocked"
    assert float(drive_cfg.drive_A) == pytest.approx(0.6)
    assert float(drive_cfg.drive_omega) == pytest.approx(1.0)
    assert float(drive_cfg.drive_tbar) == pytest.approx(1.0)
    assert str(drive_cfg.drive_pattern) == "staggered"
    assert str(drive_cfg.drive_time_sampling) == "midpoint"
    assert float(drive_cfg.drive_t0) == pytest.approx(0.0)
    assert int(drive_cfg.exact_steps_multiplier) == 1
    assert int(controller_kwargs["wallclock_cap_s"]) == 123
    assert Path(controller_kwargs["progress_path"]).name == "controller_progress.json"
    assert Path(controller_kwargs["partial_payload_path"]).name == "controller_partial.json"
    assert float(controller_kwargs["progress_every_s"]) == pytest.approx(2.0)


def test_run_adaptive_realtime_checkpoint_profile_noisy_off_runs_local_controller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--output-json",
                str(tmp_path / "noise_off.json"),
                "--output-pdf",
                str(tmp_path / "noise_off.pdf"),
                "--enable-drive",
                "--use-fake-backend",
                "--checkpoint-controller-noise-mode",
                "backend_scheduled",
                "--checkpoint-controller-mode",
                "off",
                "--checkpoint-controller-timeout-s",
                "123",
                "--checkpoint-controller-progress-every-s",
                "2.0",
            ]
        )
    )
    stage_result = _stage_result()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        noise_wf,
        "_prepare_checkpoint_controller_inputs",
        lambda stage_result_arg, staged_cfg_arg: (
            SimpleNamespace(payload_in={}),
            ScaffoldAcceptanceResult(
                accepted=True,
                reason="ok",
                structure_locked=True,
                source_kind="pytest",
            ),
            [0.1],
        ),
    )

    class _FakeController:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return SimpleNamespace(
                trajectory=[
                    {
                        "action_kind": "stay",
                        "decision_backend": "off",
                        "site_occupations": [1.0, 1.0],
                        "site_occupations_exact": [1.0, 1.0],
                        "staggered": 0.0,
                        "staggered_exact": 0.0,
                        "doublon": 0.0,
                        "doublon_exact": 0.0,
                    }
                ],
                ledger=[{"action_kind": "stay", "decision_backend": "off"}],
                summary={"mode": "off", "status": "completed", "append_count": 0, "stay_count": 1},
                reference={"kind": "driven_piecewise_constant_reference_from_replay_seed"},
            )

    monkeypatch.setattr(noise_wf, "RealtimeCheckpointController", _FakeController)

    payload = noise_wf.run_adaptive_realtime_checkpoint_profile_noisy(stage_result, cfg)

    assert payload is not None
    assert str(captured["cfg"].mode) == "off"
    assert captured["drive_config"] is not None
    assert captured["oracle_base_config"] is None
    assert int(captured["wallclock_cap_s"]) == 123
    assert Path(captured["progress_path"]).name == "controller_progress.json"
    assert Path(captured["partial_payload_path"]).name == "controller_partial.json"
    assert payload["mode"] == "off"
    assert payload["trajectory"][0]["decision_backend"] == "off"


def test_isolated_oracle_controller_nonzero_exit_returns_env_blocked_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeQueue:
        def empty(self) -> bool:
            return True

    class _FakeProc:
        exitcode = -6

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            _ = timeout
            return None

        def is_alive(self) -> bool:
            return False

        def terminate(self) -> None:
            raise AssertionError("terminate should not be called for a finished child")

    class _FakeContext:
        def Queue(self) -> _FakeQueue:
            return _FakeQueue()

        def Process(self, target, args, daemon=False) -> _FakeProc:
            assert callable(target)
            assert isinstance(args, tuple)
            assert bool(daemon) is False
            return _FakeProc()

    monkeypatch.setattr(
        noise_wf.mp,
        "get_context",
        lambda method: (_FakeContext() if str(method) == "spawn" else None),
    )

    payload = noise_wf._run_adaptive_realtime_checkpoint_profile_noisy_isolated(
        controller_kwargs={},
        mode="oracle_v1",
        scaffold_acceptance_payload={"accepted": True},
        decision_noise_mode="backend_scheduled",
        timeout_s=30,
    )

    assert payload["status"] == "env_blocked"
    assert payload["reason"] == "subprocess_nonzero_exit"
    assert payload["exitcode"] == -6
    assert payload["summary"]["status"] == "env_blocked"
    assert payload["summary"]["decision_backend"] == "env_blocked"
    assert payload["summary"]["decision_noise_mode"] == "backend_scheduled"
    assert payload["summary"]["append_count"] == 0


def test_isolated_oracle_controller_timeout_includes_progress_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    progress_path = tmp_path / "logs" / "controller_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps({"status": "running", "stage": "oracle_energy_estimate_start", "checkpoint_index": 0}),
        encoding="utf-8",
    )

    class _FakeQueue:
        def empty(self) -> bool:
            return True

    class _FakeProc:
        exitcode = None

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            _ = timeout
            return None

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            return None

    class _FakeContext:
        def Queue(self) -> _FakeQueue:
            return _FakeQueue()

        def Process(self, target, args, daemon=False) -> _FakeProc:
            assert callable(target)
            assert isinstance(args, tuple)
            assert bool(daemon) is False
            return _FakeProc()

    monkeypatch.setattr(
        noise_wf.mp,
        "get_context",
        lambda method: (_FakeContext() if str(method) == "spawn" else None),
    )

    payload = noise_wf._run_adaptive_realtime_checkpoint_profile_noisy_isolated(
        controller_kwargs={"progress_path": progress_path},
        mode="oracle_v1",
        scaffold_acceptance_payload={"accepted": True},
        decision_noise_mode="backend_scheduled",
        timeout_s=30,
    )

    assert payload["status"] == "env_blocked"
    assert payload["reason"] == "timeout_after_30s"
    assert payload["progress_snapshot"]["stage"] == "oracle_energy_estimate_start"
    assert payload["summary"]["last_progress"]["checkpoint_index"] == 0


def test_isolated_oracle_controller_timeout_preserves_partial_ledger_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    progress_path = tmp_path / "logs" / "controller_progress.json"
    partial_payload_path = tmp_path / "logs" / "controller_partial.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps({"status": "running", "stage": "checkpoint_start", "checkpoint_index": 1}),
        encoding="utf-8",
    )
    partial_payload_path.write_text(
        json.dumps(
            {
                "status": "running",
                "stage": "checkpoint_done",
                "trajectory": [{"checkpoint_index": 0, "action_kind": "append_candidate"}],
                "ledger": [
                    {
                        "checkpoint_index": 0,
                        "action_kind": "append_candidate",
                        "decision_backend": "oracle",
                        "oracle_attempted": True,
                        "logical_block_count_after": 2,
                        "runtime_parameter_count_after": 3,
                        "fidelity_exact": 0.5,
                        "abs_energy_total_error": 0.25,
                    }
                ],
                "summary": {
                    "append_count": 1,
                    "stay_count": 0,
                    "executed_decision_backends": ["oracle"],
                    "final_logical_block_count": 2,
                    "final_runtime_parameter_count": 3,
                    "final_fidelity_exact": 0.5,
                    "final_abs_energy_total_error": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )

    class _FakeQueue:
        def empty(self) -> bool:
            return True

    class _FakeProc:
        exitcode = None

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            _ = timeout
            return None

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            return None

    class _FakeContext:
        def Queue(self) -> _FakeQueue:
            return _FakeQueue()

        def Process(self, target, args, daemon=False) -> _FakeProc:
            assert callable(target)
            assert isinstance(args, tuple)
            assert bool(daemon) is False
            return _FakeProc()

    monkeypatch.setattr(
        noise_wf.mp,
        "get_context",
        lambda method: (_FakeContext() if str(method) == "spawn" else None),
    )

    payload = noise_wf._run_adaptive_realtime_checkpoint_profile_noisy_isolated(
        controller_kwargs={
            "progress_path": progress_path,
            "partial_payload_path": partial_payload_path,
        },
        mode="oracle_v1",
        scaffold_acceptance_payload={"accepted": True},
        decision_noise_mode="backend_scheduled",
        timeout_s=30,
    )

    assert payload["status"] == "env_blocked"
    assert payload["reason"] == "timeout_after_30s"
    assert len(payload["trajectory"]) == 1
    assert len(payload["ledger"]) == 1
    assert payload["summary"]["append_count"] == 1
    assert payload["summary"]["oracle_decision_checkpoints"] == 1
    assert payload["summary"]["final_logical_block_count"] == 2


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
                "--fixed-scaffold-runtime-transpile-optimization-level",
                "2",
                "--fixed-scaffold-runtime-seed-transpiler",
                "5",
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
    seen_ansatz_kwargs: dict[str, object] = {}

    def _fake_ansatz_audit_mode(source_cfg, noise_cfg, mode, **kwargs):
        seen_ansatz_kwargs.update(kwargs)
        return {"success": True, "kind": "ansatz_input", "mode": mode}

    monkeypatch.setattr(
        noise_wf,
        "_run_imported_ansatz_input_state_audit_mode",
        _fake_ansatz_audit_mode,
    )
    seen_full_kwargs: dict[str, object] = {}

    def _fake_full_audit_mode(source_cfg, noise_cfg, mode, **kwargs):
        seen_full_kwargs.update(kwargs)
        return {"success": True, "kind": "full", "mode": mode}

    monkeypatch.setattr(
        noise_wf,
        "_run_imported_full_circuit_audit_mode",
        _fake_full_audit_mode,
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
    assert payload["imported_ansatz_input_state_audit"]["modes"]["ideal"]["success"] is True
    assert payload["full_circuit_import_audit"]["modes"]["backend_scheduled"]["success"] is True
    assert seen_ansatz_kwargs["seed_transpiler"] == 5
    assert seen_ansatz_kwargs["transpile_optimization_level"] == 2
    assert seen_ansatz_kwargs["compile_request_source"] == "fixed_scaffold_runtime_transpile_cli"
    assert seen_full_kwargs["seed_transpiler"] == 5
    assert seen_full_kwargs["transpile_optimization_level"] == 2
    assert seen_full_kwargs["compile_request_source"] == "fixed_scaffold_runtime_transpile_cli"
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
        "_run_imported_ansatz_input_state_audit_mode",
        lambda source_cfg, noise_cfg, mode, **kwargs: (_ for _ in ()).throw(
            AssertionError("ansatz-input-state audit should stay disabled without --include-full-circuit-audit")
        ),
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_full_circuit_audit_mode",
        lambda source_cfg, noise_cfg, mode, **kwargs: {"success": True, "kind": "full", "mode": mode},
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
    assert payload["workflow_contract"]["imported_routes"]["ansatz_input_state_audit"] is False
    assert payload["settings"]["fixed_lean_noisy_replay"]["enabled"] is True
    assert payload["settings"]["fixed_lean_noise_attribution"]["enabled"] is False
    assert payload["imported_ansatz_input_state_audit"]["modes"] == {}
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
        lambda source_cfg, noise_cfg, mode, **kwargs: {"success": True, "kind": "full", "mode": mode},
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


def test_run_staged_hh_noise_import_mode_preserves_partial_fixed_scaffold_noisy_replay_timeout(
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
                str(tmp_path / "import_noise_timeout.json"),
                "--output-pdf",
                str(tmp_path / "import_noise_timeout.pdf"),
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
        lambda source_cfg, noise_cfg, mode, **kwargs: {"success": True, "kind": "full", "mode": mode},
    )
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noise_attribution_mode", lambda source_cfg, noise_cfg, attribution_cfg: {})
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_scaffold_noisy_replay_mode",
        lambda source_cfg, noise_cfg, fixed_cfg: {
            "success": False,
            "env_blocked": True,
            "partial": True,
            "route": "fixed_scaffold_noisy_replay",
            "reason": "timeout_after_600s",
            "theta_source": "imported_theta_runtime",
            "execution_mode": "backend_scheduled",
            "local_mitigation_label": "readout_only",
            "objective_trace": [{"call_index": 1, "energy_noisy_mean": 0.4}],
            "best_so_far": {"call_index": 1, "energy_noisy_mean": 0.4},
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

    payload = noise_wf.run_staged_hh_noise(
        cfg,
        run_command="python pipelines/hardcoded/hh_staged_noise.py --include-fixed-scaffold-noisy-replay",
    )

    assert payload["fixed_scaffold_noisy_replay"]["partial"] is True
    assert payload["fixed_scaffold_noisy_replay"]["reason"] == "timeout_after_600s"
    assert payload["summary"]["fixed_scaffold_noisy_replay_completed"] == 0
    assert payload["summary"]["fixed_scaffold_noisy_replay_total"] == 1
    assert payload["summary"]["fixed_scaffold_noisy_replay_execution_mode"] == "backend_scheduled"
    assert np.isnan(payload["summary"]["fixed_scaffold_best_noisy_minus_ideal"])
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
        lambda source_cfg, noise_cfg, mode, **kwargs: {"success": True, "kind": "full", "mode": mode},
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


def test_run_staged_hh_noise_import_mode_runs_fixed_scaffold_runtime_raw_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "runtime_fixed_scaffold.json"
    source_json.write_text(
        "{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}",
        encoding="utf-8",
    )
    raw_artifact_path = tmp_path / "raw_records.ndjson.gz"
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-scaffold-runtime-raw-baseline",
                "--fixed-final-state-json",
                str(source_json),
                "--backend-name",
                "ibm_marrakesh",
                "--fixed-scaffold-runtime-raw-profile",
                "raw_sampler_twirled_v1",
                "--symmetry-mitigation-mode",
                "verify_only",
                "--fixed-scaffold-runtime-raw-transport",
                "sampler_v2",
                "--fixed-scaffold-runtime-raw-store-memory",
                "--fixed-scaffold-runtime-raw-artifact-path",
                str(raw_artifact_path),
                "--output-json",
                str(tmp_path / "runtime_raw_baseline.json"),
                "--output-pdf",
                str(tmp_path / "runtime_raw_baseline.pdf"),
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
        lambda source_cfg, noise_cfg, mode, **kwargs: {"success": True, "kind": "full", "mode": mode},
    )
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noise_attribution_mode", lambda source_cfg, noise_cfg, attribution_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_compile_control_scout_mode", lambda source_cfg, noise_cfg, scout_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_compile_control_scout_mode", lambda source_cfg, noise_cfg, scout_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_noise_attribution_mode", lambda source_cfg, noise_cfg, attribution_cfg: {})
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_scaffold_runtime_raw_baseline_mode",
        lambda source_cfg, noise_cfg, runtime_cfg: {
            "success": True,
            "route": "fixed_scaffold_runtime_raw_baseline",
            "subject_kind": "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1",
            "energy_audits": {"main": {"success": True, "evaluation": {"delta_mean": 0.123}}},
            "noise_config": {
                "noise_mode": "runtime",
                "symmetry_mitigation": {"mode": "verify_only"},
                "runtime_profile": {"name": "raw_sampler_twirled_v1"},
                "execution_surface": "raw_measurement_v1",
                "raw_transport": "sampler_v2",
                "raw_store_memory": True,
                "raw_artifact_path": str(raw_artifact_path),
            },
            "compile_control": {
                "backend_name": "ibm_marrakesh",
                "seed_transpiler": 0,
                "transpile_optimization_level": 1,
                "source": "fixed_scaffold_runtime_transpile_cli",
            },
            "compile_observation": {
                "available": True,
                "requested": {
                    "backend_name": "ibm_marrakesh",
                    "seed_transpiler": 0,
                    "transpile_optimization_level": 1,
                    "source": "fixed_scaffold_runtime_transpile_cli",
                },
                "observed": {
                    "backend_name": "ibm_marrakesh",
                    "seed_transpiler": 0,
                    "transpile_optimization_level": 1,
                },
                "matches_requested": True,
                "mismatch_fields": [],
                "reason": None,
            },
            "backend_info": {
                "details": {
                    "execution_surface": "raw_measurement_v1",
                    "transport": "sampler_v2",
                    "raw_artifact_path": str(raw_artifact_path),
                }
            },
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
        run_command="python pipelines/hardcoded/hh_staged_noise.py --include-fixed-scaffold-runtime-raw-baseline --backend-name ibm_marrakesh",
    )

    assert payload["workflow_contract"]["imported_routes"]["fixed_scaffold_runtime_raw_baseline"] is True
    assert payload["settings"]["fixed_scaffold_runtime_raw_baseline"]["enabled"] is True
    assert payload["fixed_scaffold_runtime_raw_baseline"]["success"] is True
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_completed"] == 1
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_main_delta_mean"] == pytest.approx(0.123)
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_noise_mode"] == "runtime"
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_execution_surface"] == "raw_measurement_v1"
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_profile_name"] == "raw_sampler_twirled_v1"
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_symmetry_mode"] == "verify_only"
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_raw_transport"] == "sampler_v2"
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_raw_store_memory"] is True
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_raw_artifact_path"] == str(
        raw_artifact_path
    )
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_diagonal_postprocessing_available"] is False
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_requested_seed_transpiler"] == 0
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_requested_transpile_optimization_level"] == 1
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_observed_seed_transpiler"] == 0
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_observed_transpile_optimization_level"] == 1
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_compile_request_matched"] is True
    assert payload["artifacts"]["fixed_scaffold_runtime_raw_baseline_json"] is not None
    sidecar_name = Path(payload["artifacts"]["fixed_scaffold_runtime_raw_baseline_json"]).name
    assert sidecar_name in writes
    sidecar_payload = writes[sidecar_name]
    assert sidecar_payload["pipeline"] == "hh_fixed_scaffold_runtime_raw_baseline_eval_v1"
    assert sidecar_payload["backend_name"] == "ibm_marrakesh"
    assert (
        sidecar_payload["settings"]["runtime_raw_baseline"]["raw_transport"] == "sampler_v2"
    )
    assert sidecar_payload["settings"]["runtime_raw_baseline"]["raw_store_memory"] is True
    assert (
        sidecar_payload["settings"]["runtime_raw_baseline"]["raw_artifact_path"]
        == str(raw_artifact_path)
    )
    assert (
        sidecar_payload["settings"]["runtime_raw_baseline"]["runtime_profile_config"]["name"]
        == "raw_sampler_twirled_v1"
    )
    assert (
        sidecar_payload["settings"]["runtime_raw_baseline"]["symmetry_mitigation_config"]["mode"]
        == "verify_only"
    )
    assert bool(writes["pdf"]) is True


def test_run_staged_hh_noise_import_mode_runs_fixed_scaffold_runtime_raw_baseline_local_postprocessing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "runtime_fixed_scaffold.json"
    source_json.write_text(
        "{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}",
        encoding="utf-8",
    )
    raw_artifact_path = tmp_path / "local_raw_records.ndjson.gz"
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-scaffold-runtime-raw-baseline",
                "--fixed-final-state-json",
                str(source_json),
                "--use-fake-backend",
                "--mitigation",
                "readout",
                "--symmetry-mitigation-mode",
                "projector_renorm_v1",
                "--fixed-scaffold-runtime-raw-artifact-path",
                str(raw_artifact_path),
                "--output-json",
                str(tmp_path / "runtime_raw_baseline_local.json"),
                "--output-pdf",
                str(tmp_path / "runtime_raw_baseline_local.pdf"),
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
        lambda source_cfg, noise_cfg, mode, **kwargs: {},
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_imported_full_circuit_audit_mode",
        lambda source_cfg, noise_cfg, mode, **kwargs: {},
    )
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_noise_attribution_mode", lambda source_cfg, noise_cfg, attribution_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_lean_compile_control_scout_mode", lambda source_cfg, noise_cfg, scout_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_noisy_replay_mode", lambda source_cfg, noise_cfg, fixed_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_compile_control_scout_mode", lambda source_cfg, noise_cfg, scout_cfg: {})
    monkeypatch.setattr(noise_wf, "_run_fixed_scaffold_noise_attribution_mode", lambda source_cfg, noise_cfg, attribution_cfg: {})
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_scaffold_runtime_raw_baseline_mode",
        lambda source_cfg, noise_cfg, runtime_cfg: {
            "success": True,
            "route": "fixed_scaffold_runtime_raw_baseline",
            "subject_kind": "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1",
            "energy_audits": {
                "main": {
                    "success": True,
                    "evaluation": {"delta_mean": 0.045},
                    "diagonal_postprocessing": {"available": True},
                }
            },
            "noise_config": {
                "noise_mode": "backend_scheduled",
                "execution_surface": "raw_measurement_v1",
                "raw_transport": "auto",
                "raw_store_memory": False,
                "raw_artifact_path": str(raw_artifact_path),
            },
            "compile_control": {
                "backend_name": "FakeMarrakesh",
                "seed_transpiler": 0,
                "transpile_optimization_level": 1,
                "source": "fixed_scaffold_runtime_transpile_cli",
            },
            "compile_observation": {
                "available": True,
                "requested": {
                    "backend_name": "FakeMarrakesh",
                    "seed_transpiler": 0,
                    "transpile_optimization_level": 1,
                    "source": "fixed_scaffold_runtime_transpile_cli",
                },
                "observed": {
                    "backend_name": "FakeMarrakesh",
                    "seed_transpiler": 2,
                    "transpile_optimization_level": 1,
                },
                "matches_requested": False,
                "mismatch_fields": ["seed_transpiler"],
                "reason": None,
            },
            "backend_info": {
                "details": {
                    "execution_surface": "raw_measurement_v1",
                    "transport": "backend_run",
                    "raw_artifact_path": str(raw_artifact_path),
                }
            },
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
        run_command="python pipelines/hardcoded/hh_staged_noise.py --include-fixed-scaffold-runtime-raw-baseline --use-fake-backend --mitigation readout --symmetry-mitigation-mode projector_renorm_v1",
    )

    assert payload["fixed_scaffold_runtime_raw_baseline"]["success"] is True
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_noise_mode"] == "backend_scheduled"
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_raw_transport"] == "backend_run"
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_diagonal_postprocessing_available"] is True
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_requested_seed_transpiler"] == 0
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_observed_seed_transpiler"] == 2
    assert payload["summary"]["fixed_scaffold_runtime_raw_baseline_compile_request_matched"] is False
    sidecar_name = Path(payload["artifacts"]["fixed_scaffold_runtime_raw_baseline_json"]).name
    sidecar_payload = writes[sidecar_name]
    assert sidecar_payload["backend_name"] == "FakeMarrakesh"
    assert sidecar_payload["settings"]["runtime_raw_baseline"]["noise_mode"] == "backend_scheduled"
    assert sidecar_payload["settings"]["runtime_raw_baseline"]["mitigation_config"]["mode"] == "readout"
    assert sidecar_payload["settings"]["runtime_raw_baseline"]["symmetry_mitigation_config"]["mode"] == "projector_renorm_v1"


def test_run_staged_hh_noise_import_mode_runs_fixed_scaffold_runtime_pairing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_json = tmp_path / "runtime_pairing.json"
    source_json.write_text(
        "{\"adapt_vqe\": {\"operators\": []}, \"ground_state\": {}, \"settings\": {}}",
        encoding="utf-8",
    )
    raw_artifact_path = tmp_path / "paired_raw_records.ndjson.gz"
    cfg = noise_wf.resolve_staged_hh_noise_config(
        parse_args(
            [
                "--L",
                "2",
                "--skip-pdf",
                "--include-fixed-scaffold-runtime-energy-only-baseline",
                "--include-fixed-scaffold-runtime-raw-baseline",
                "--fixed-final-state-json",
                str(source_json),
                "--backend-name",
                "ibm_marrakesh",
                "--include-fixed-scaffold-runtime-final-zne-audit",
                "--fixed-scaffold-runtime-raw-profile",
                "raw_sampler_twirled_v1",
                "--symmetry-mitigation-mode",
                "verify_only",
                "--fixed-scaffold-runtime-raw-transport",
                "sampler_v2",
                "--fixed-scaffold-runtime-raw-artifact-path",
                str(raw_artifact_path),
                "--output-json",
                str(tmp_path / "runtime_pairing.json"),
                "--output-pdf",
                str(tmp_path / "runtime_pairing.pdf"),
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
        lambda source_cfg, noise_cfg, mode, **kwargs: {"success": True, "kind": "full", "mode": mode},
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
            "noise_config": {
                "noise_mode": "runtime",
                "backend_name": "ibm_marrakesh",
                "runtime_profile": {"name": "main_twirled_readout_v1"},
            },
            "energy_audits": {
                "main": {"success": True, "evaluation": {"delta_mean": 0.234}},
                "final_audit_zne": {"success": True, "evaluation": {"delta_mean": 0.111}},
            },
        },
    )
    monkeypatch.setattr(
        noise_wf,
        "_run_fixed_scaffold_runtime_raw_baseline_mode",
        lambda source_cfg, noise_cfg, runtime_cfg: {
            "success": True,
            "route": "fixed_scaffold_runtime_raw_baseline",
            "subject_kind": "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1",
            "energy_audits": {"main": {"success": True, "evaluation": {"delta_mean": 0.123}}},
            "noise_config": {
                "noise_mode": "runtime",
                "backend_name": "ibm_marrakesh",
                "symmetry_mitigation": {"mode": "verify_only"},
                "runtime_profile": {"name": "raw_sampler_twirled_v1"},
                "execution_surface": "raw_measurement_v1",
                "raw_transport": "sampler_v2",
                "raw_store_memory": False,
                "raw_artifact_path": str(raw_artifact_path),
            },
            "compile_control": {
                "backend_name": "ibm_marrakesh",
                "seed_transpiler": 0,
                "transpile_optimization_level": 1,
                "source": "fixed_scaffold_runtime_transpile_cli",
            },
            "compile_observation": {
                "available": True,
                "requested": {
                    "backend_name": "ibm_marrakesh",
                    "seed_transpiler": 0,
                    "transpile_optimization_level": 1,
                    "source": "fixed_scaffold_runtime_transpile_cli",
                },
                "observed": {
                    "backend_name": "ibm_marrakesh",
                    "seed_transpiler": 0,
                    "transpile_optimization_level": 1,
                },
                "matches_requested": True,
                "mismatch_fields": [],
                "reason": None,
            },
            "backend_info": {
                "details": {
                    "execution_surface": "raw_measurement_v1",
                    "transport": "sampler_v2",
                    "raw_artifact_path": str(raw_artifact_path),
                }
            },
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
        run_command=(
            "python pipelines/hardcoded/hh_staged_noise.py "
            "--include-fixed-scaffold-runtime-energy-only-baseline "
            "--include-fixed-scaffold-runtime-raw-baseline --backend-name ibm_marrakesh"
        ),
    )

    assert payload["workflow_contract"]["imported_routes"]["fixed_scaffold_runtime_energy_only"] is True
    assert payload["workflow_contract"]["imported_routes"]["fixed_scaffold_runtime_raw_baseline"] is True
    assert payload["workflow_contract"]["imported_routes"]["fixed_scaffold_runtime_pairing"] is True
    assert payload["fixed_scaffold_runtime_pairing"]["success"] is True
    assert payload["fixed_scaffold_runtime_pairing"]["requested_compile_match"] is True
    assert payload["fixed_scaffold_runtime_pairing"]["raw_compile_observation_matches_requested"] is True
    assert payload["fixed_scaffold_runtime_pairing"]["energy_audit_labels"] == [
        "final_audit_zne",
        "main",
    ]
    assert payload["summary"]["fixed_scaffold_runtime_pairing_completed"] == 1
    assert payload["summary"]["fixed_scaffold_runtime_pairing_total"] == 1
    assert payload["summary"]["fixed_scaffold_runtime_pairing_subject_kind_match"] is True
    assert payload["summary"]["fixed_scaffold_runtime_pairing_backend_name_match"] is True
    assert payload["summary"]["fixed_scaffold_runtime_pairing_requested_compile_match"] is True
    assert payload["summary"]["fixed_scaffold_runtime_pairing_raw_compile_observation_matched"] is True
    assert payload["summary"]["fixed_scaffold_runtime_pairing_energy_audit_labels"] == "final_audit_zne,main"
    assert payload["summary"]["fixed_scaffold_runtime_pairing_raw_artifact_path"] == str(
        raw_artifact_path
    )
    assert payload["artifacts"]["fixed_scaffold_runtime_pairing_json"] is not None
    pairing_sidecar_name = Path(payload["artifacts"]["fixed_scaffold_runtime_pairing_json"]).name
    assert pairing_sidecar_name in writes
    pairing_sidecar_payload = writes[pairing_sidecar_name]
    assert pairing_sidecar_payload["pipeline"] == "hh_fixed_scaffold_runtime_pairing_eval_v1"
    assert pairing_sidecar_payload["backend_name"] == "ibm_marrakesh"
    assert pairing_sidecar_payload["result"]["raw_transport"] == "sampler_v2"
    assert (
        pairing_sidecar_payload["settings"]["runtime_energy_only"]["include_final_zne_audit"]
        is True
    )
    assert (
        pairing_sidecar_payload["settings"]["runtime_raw_baseline"]["runtime_profile_config"]["name"]
        == "raw_sampler_twirled_v1"
    )
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
        lambda source_cfg, noise_cfg, mode, **kwargs: {"success": True, "kind": "full", "mode": mode},
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
        lambda source_cfg, noise_cfg, mode, **kwargs: {"success": True, "kind": "full", "mode": mode},
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
        lambda source_cfg, noise_cfg, mode, **kwargs: {"success": True, "kind": "full", "mode": mode},
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
                "FakeMarrakesh",
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
        lambda source_cfg, noise_cfg, mode, **kwargs: {"success": True, "kind": "full", "mode": mode},
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
            "subject_kind": "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1",
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
        run_command="python pipelines/hardcoded/hh_staged_noise.py --include-fixed-scaffold-compile-control-scout --backend-name FakeMarrakesh",
    )

    assert payload["workflow_contract"]["imported_routes"]["fixed_scaffold_compile_control_scout"] is True
    assert payload["settings"]["fixed_scaffold_compile_control_scout"]["enabled"] is True
    assert payload["fixed_scaffold_compile_control_scout"]["success"] is True
    assert payload["summary"]["fixed_scaffold_compile_control_scout_completed"] == 1
    assert payload["summary"]["fixed_scaffold_compile_control_scout_candidates_successful"] == 8
    assert payload["summary"]["fixed_scaffold_compile_control_scout_best_two_qubit_count"] == pytest.approx(19.0)
    assert writes["path"] == cfg.staged.artifacts.output_json


def test_build_noise_summary_preserves_partial_fixed_scaffold_compile_control_timeout() -> None:
    summary = noise_wf._build_noise_summary(
        {
            "fixed_scaffold_compile_control_scout": {
                "success": False,
                "reason": "timeout_after_1200s",
                "candidate_counts": {
                    "total": 10,
                    "completed": 3,
                    "successful": 2,
                    "failed": 1,
                },
                "best_candidate": {
                    "label": "opt2_seed0",
                    "delta_mean": 0.31,
                    "compiled_two_qubit_count": 19,
                    "compiled_depth": 57,
                },
            }
        }
    )

    assert summary["fixed_scaffold_compile_control_scout_completed"] == 0
    assert summary["fixed_scaffold_compile_control_scout_total"] == 1
    assert summary["fixed_scaffold_compile_control_scout_candidates_total"] == 10
    assert summary["fixed_scaffold_compile_control_scout_candidates_successful"] == 2
    assert summary["fixed_scaffold_compile_control_scout_best_delta_mean"] == pytest.approx(0.31)
    assert summary["fixed_scaffold_compile_control_scout_best_two_qubit_count"] == pytest.approx(19.0)
    assert summary["fixed_scaffold_compile_control_scout_best_depth"] == pytest.approx(57.0)
    assert "fixed_scaffold_compile_control_scout:timeout_after_1200s" in summary["failure_samples"]


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
            "fixed_scaffold_runtime_raw_baseline_json": "artifacts/json/hh_staged_noise_raw_baseline.json",
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
        "fixed_scaffold_runtime_raw_baseline_json=artifacts/json/hh_staged_noise_raw_baseline.json",
        "adapt_handoff_json=artifacts/json/hh_staged_noise_adapt_handoff.json",
    ]


def test_noise_cli_main_print_contract_import_mode_runtime_energy_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_cfg = SimpleNamespace(staged=SimpleNamespace(artifacts=SimpleNamespace(skip_pdf=True)))
    fake_payload = {
        "artifacts": {
            "workflow": {
                "output_json": "artifacts/json/hh_staged_noise.json",
                "output_pdf": "artifacts/pdf/hh_staged_noise.pdf",
            },
            "import_source_json": "artifacts/json/adapt_hh_L2.json",
            "fixed_scaffold_runtime_energy_only_json": "artifacts/json/hh_staged_noise_runtime_energy_only.json",
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
        "fixed_scaffold_runtime_energy_only_json=artifacts/json/hh_staged_noise_runtime_energy_only.json",
        "adapt_handoff_json=artifacts/json/hh_staged_noise_adapt_handoff.json",
    ]


def test_noise_cli_main_print_contract_import_mode_runtime_pairing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_cfg = SimpleNamespace(staged=SimpleNamespace(artifacts=SimpleNamespace(skip_pdf=True)))
    fake_payload = {
        "artifacts": {
            "workflow": {
                "output_json": "artifacts/json/hh_staged_noise.json",
                "output_pdf": "artifacts/pdf/hh_staged_noise.pdf",
            },
            "import_source_json": "artifacts/json/adapt_hh_L2.json",
            "fixed_scaffold_runtime_energy_only_json": "artifacts/json/hh_staged_noise_runtime_energy_only.json",
            "fixed_scaffold_runtime_raw_baseline_json": "artifacts/json/hh_staged_noise_runtime_raw_baseline.json",
            "fixed_scaffold_runtime_pairing_json": "artifacts/json/hh_staged_noise_runtime_pairing.json",
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
        "fixed_scaffold_runtime_energy_only_json=artifacts/json/hh_staged_noise_runtime_energy_only.json",
        "fixed_scaffold_runtime_raw_baseline_json=artifacts/json/hh_staged_noise_runtime_raw_baseline.json",
        "fixed_scaffold_runtime_pairing_json=artifacts/json/hh_staged_noise_runtime_pairing.json",
        "adapt_handoff_json=artifacts/json/hh_staged_noise_adapt_handoff.json",
    ]
