from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.time_dynamics.runners.hh_from_adapt_artifact import (
    build_controller_config,
    build_parser as build_realtime_parser,
)
from pipelines.time_dynamics.legacy.checkpoint_types import RealtimeCheckpointConfig
from pipelines.time_dynamics.legacy.analysis.hh_time_dynamics_spectra import render_spectrum_pdf
from pipelines.time_dynamics.runners.generic_from_adapt_artifact import (
    build_parser as build_generic_realtime_parser,
)
import pipelines.time_dynamics.optimization.hh_realtime_optuna as realtime_optuna
from pipelines.time_dynamics.optimization.hh_realtime_optuna import (
    BaseRunConfig,
    ObjectiveWeights,
    TrialParams,
    ValidityGates,
    _apply_base_run_trial_overrides,
    _baseline_trial_params,
    _build_realtime_tokens,
    _evaluate_trial,
    _invalid_reasons,
    _objective_value,
    _profile_seed_trials,
    _resolved_pair,
    _search_space,
    _trial_metrics_from_payload,
    _write_json,
)

_PREFSITE_AUTOBASELINE_PROFILES = (
    (
        "high_amp_guarded_d_shape_site3_prefsite_autobaseline_v1",
        "d_shape_barrier_v1",
    ),
    (
        "high_amp_guarded_fidelity_first_site3_prefsite_autobaseline_v1",
        "fidelity_first_barrier_v1",
    ),
)

_FULL_SURFACE_PROFILE = "high_amp_guarded_site3_full_surface_v2"


def _synthetic_payload() -> dict[str, object]:
    trajectory = [
        {
            "time": 0.0,
            "physical_time": 0.0,
            "site_occupations": [1.0, 1.0],
            "site_occupations_exact": [1.0, 1.0],
            "staggered": 0.0,
            "staggered_exact": 0.0,
            "doublon": 0.1,
            "doublon_exact": 0.1,
            "energy_total": 0.0,
            "energy_total_controller": 0.0,
            "energy_total_exact": 0.0,
            "abs_energy_total_error": 0.0,
            "fidelity_exact": 1.0,
            "site_occupations_abs_error_max": 0.0,
            "runtime_parameter_count": 2,
            "tracking_d_curvature_abs_error_mean": 0.0,
            "tracking_d_excursion_under_response_mean": 0.0,
            "tracking_d_excursion_over_response_mean": 0.0,
            "tracking_total_occupation_abs_error_mean": 0.0,
        },
        {
            "time": 1.0,
            "physical_time": 1.0,
            "site_occupations": [1.2, 0.8],
            "site_occupations_exact": [1.1, 0.9],
            "staggered": 0.2,
            "staggered_exact": 0.1,
            "doublon": 0.2,
            "doublon_exact": 0.15,
            "energy_total": 0.3,
            "energy_total_controller": 0.3,
            "energy_total_exact": 0.2,
            "abs_energy_total_error": 0.1,
            "fidelity_exact": 0.95,
            "site_occupations_abs_error_max": 0.1,
            "runtime_parameter_count": 3,
            "tracking_d_curvature_abs_error_mean": 0.2,
            "tracking_d_excursion_under_response_mean": 0.3,
            "tracking_d_excursion_over_response_mean": 0.1,
            "tracking_total_occupation_abs_error_mean": 0.0,
        },
        {
            "time": 2.0,
            "physical_time": 2.0,
            "site_occupations": [0.7, 1.3],
            "site_occupations_exact": [0.9, 1.1],
            "staggered": -0.3,
            "staggered_exact": -0.1,
            "doublon": 0.3,
            "doublon_exact": 0.2,
            "energy_total": 0.4,
            "energy_total_controller": 0.4,
            "energy_total_exact": 0.25,
            "abs_energy_total_error": 0.15,
            "fidelity_exact": 0.9,
            "site_occupations_abs_error_max": 0.2,
            "runtime_parameter_count": 4,
            "tracking_d_curvature_abs_error_mean": 0.4,
            "tracking_d_excursion_under_response_mean": 0.6,
            "tracking_d_excursion_over_response_mean": 0.2,
            "tracking_total_occupation_abs_error_mean": 0.0,
        },
    ]
    return {
        "run_tag": "synthetic_trial",
        "summary": {
            "append_count": 1,
            "final_runtime_parameter_count": 4,
            "max_abs_energy_total_error": 0.15,
            "max_abs_site_occupations_error": 0.2,
            "oracle_compile_observation": {
                "compiled_count_2q": 42,
                "compiled_depth": 99,
                "compiled_size": 123,
                "compiled_num_qubits": 6,
            },
            "oracle_backend_snapshot": {"backend_name": "FakeMarrakesh"},
            "oracle_compile_request": {
                "transpile_seed": 7,
                "transpile_optimization_level": 2,
            },
        },
        "trajectory": trajectory,
        "reference": {
            "drive_profile": {
                "drive_omega": 1.0,
                "drive_A": 0.5,
            }
        },
    }


def _short_early_stop_payload() -> dict[str, object]:
    payload = _synthetic_payload()
    row = dict(list(payload["trajectory"])[0])
    row.update(
        {
            "abs_energy_total_error": 0.25,
            "site_occupations_abs_error_max": 0.125,
            "fidelity_exact": 0.8,
            "runtime_parameter_count": 2,
            "exact_v1_selection_reason": "high_miss_no_admit_repair_stop",
            "proposed_action_kind": "stay",
        }
    )
    return {
        "run_tag": "short_repair_stop_trial",
        "summary": {
            "status": "repair_stop",
            "early_stop_reason": "high_miss_no_admit_repair_stop",
            "append_count": 0,
            "final_runtime_parameter_count": 2,
            "max_abs_energy_total_error": 0.25,
            "max_abs_site_occupations_error": 0.125,
        },
        "trajectory": [row],
        "reference": payload["reference"],
    }


def _strict_synthetic_payload(
    *,
    family: str = "hh",
    strict_qpu_hh: bool | None = None,
) -> dict[str, object]:
    strict_hh = bool(family == "hh") if strict_qpu_hh is None else bool(strict_qpu_hh)
    trajectory: list[dict[str, object]] = []
    for idx, (time_value, rho_miss, runtime_count) in enumerate(
        [(0.0, 0.02, 2), (1.0, 0.04, 3), (2.0, 0.01, 3)]
    ):
        trajectory.append(
            {
                "checkpoint_index": idx,
                "time": time_value,
                "physical_time": time_value,
                "trajectory_sample_kind": "state_sample",
                "advances_time": True,
                "action_kind": "stay",
                "proposed_action_kind": "stay",
                "controller_lane": "stay",
                "decision_backend": "oracle",
                "decision_noise_mode": "ideal",
                "oracle_attempted": True,
                "oracle_decision_used": True,
                "decision_path_kind": "strict_qpu_faithful_observable_v1",
                "strict_qpu_faithful": True,
                "strict_qpu_hh": strict_hh,
                "strict_qpu_family": str(family),
                "integrator_policy": "euler",
                "integrator_used": "none" if idx == 2 else "euler",
                "rho_miss": rho_miss,
                "rho_real": rho_miss,
                "rho_num": 0.0,
                "predicted_displacement": 0.005 * (idx + 1),
                "runtime_parameter_count": runtime_count,
                "logical_block_count": runtime_count,
                "selected_noisy_improvement_abs": 0.0,
                "selected_noisy_improvement_ratio": 0.0,
                "degraded_reason": None,
                # Exact/reference-only payload fields must be ignored by strict scoring.
                "energy_total_exact": 123.0 + idx,
                "abs_energy_total_error": 456.0 + idx,
                "fidelity_exact": 0.01 * idx,
                "site_occupations_exact": [999.0, -999.0],
                "site_occupations_abs_error_max": 777.0,
            }
        )
    return {
        "run_tag": "strict_synthetic_trial",
        "route_config": {
            "strict_qpu_faithful": True,
            "strict_qpu_hh": strict_hh,
            "problem_family": str(family),
        },
        "summary": {
            "mode": "oracle_v1",
            "reference_mode": "off",
            "reference_enabled": False,
            "decision_path_kind": "strict_qpu_faithful_observable_v1",
            "strict_qpu_faithful": True,
            "strict_qpu_hh": strict_hh,
            "strict_qpu_family": str(family),
            "strict_fail_closed": False,
            "strict_fail_closed_reason": None,
            "qpu_faithful_decisions_expected": True,
            "qpu_faithful_decisions_passed": True,
            "status": "completed",
            "decision_noise_mode": "ideal",
            "exact_decision_checkpoints": 0,
            "oracle_decision_checkpoints": 3,
            "append_count": 0,
            "prune_count": 0,
            "high_miss_count": 0,
            "high_miss_fraction": 0.0,
            "high_miss_no_admit_count": 0,
            "high_miss_no_admit_fraction": 0.0,
            "high_miss_no_admit_soft_fallback_count": 0,
            "degraded_checkpoints": 0,
            "final_runtime_parameter_count": 3,
            "final_abs_energy_total_error": None,
            "final_fidelity_exact": None,
        },
        "trajectory": trajectory,
        "ledger": [dict(row) for row in trajectory],
        "reference": {
            "reference_mode": "off",
            "reference_enabled": False,
            "kind": None,
        },
    }


def test_resolved_pair_auto_uses_l2_default() -> None:
    assert _resolved_pair("auto", num_sites=2) == (0, 1)
    assert _resolved_pair("none", num_sites=2) is None
    assert _resolved_pair("1,2", num_sites=4) == (1, 2)


def test_build_realtime_tokens_explicit_repair_retry_flags_round_trip() -> None:
    tokens = _build_realtime_tokens(
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json")),
        params=TrialParams(
            high_miss_no_admit_policy="repair_retry",
            repair_retry_max_attempts=1,
            repair_retry_escalation_mode="append_budget_then_stabilize_v1",
            repair_retry_admission_policy="rescue_best_confirmed_append_v1",
            repair_retry_rescue_min_gain_ratio=0.125,
            repair_retry_rescue_attempt="terminal_attempt_only",
        ),
        output_json=Path("out.json"),
        run_tag="retry_roundtrip",
    )
    args = build_realtime_parser().parse_args(tokens)
    cfg = build_controller_config(args)

    assert cfg.high_miss_no_admit_policy == "repair_retry"
    assert cfg.repair_retry_max_attempts == 1
    assert cfg.repair_retry_escalation_mode == "append_budget_then_stabilize_v1"
    assert cfg.repair_retry_admission_policy == "rescue_best_confirmed_append_v1"
    assert cfg.repair_retry_rescue_min_gain_ratio == pytest.approx(0.125)
    assert cfg.repair_retry_rescue_attempt == "terminal_attempt_only"


def test_build_realtime_tokens_normalizes_legacy_high_miss_alias() -> None:
    tokens = _build_realtime_tokens(
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json")),
        params=TrialParams(high_miss_no_admit_policy="legacy_advance_stay"),
        output_json=Path("out.json"),
        run_tag="alias_roundtrip",
    )
    args = build_realtime_parser().parse_args(tokens)
    cfg = build_controller_config(args)

    assert tokens[tokens.index("--checkpoint-controller-high-miss-no-admit-policy") + 1] == "bounded_stay_advance"
    assert cfg.high_miss_no_admit_policy == "bounded_stay_advance"


def test_base_run_integrator_policy_override_forces_trial_policy() -> None:
    params = TrialParams(integrator_policy="auto_euler_rk4")
    base = BaseRunConfig(
        artifact_json=Path("artifact.json"),
        integrator_policy_override="rk4",
    )
    forced = _apply_base_run_trial_overrides(params, base)

    assert params.integrator_policy == "auto_euler_rk4"
    assert forced.integrator_policy == "rk4"
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=forced,
        output_json=Path("out.json"),
        run_tag="forced_rk4",
    )
    assert tokens[tokens.index("--checkpoint-controller-integrator-policy") + 1] == "rk4"


def test_search_spaces_use_canonical_bounded_policy_not_legacy_alias() -> None:
    for profile in ("generic_l2_exact_v1", _FULL_SURFACE_PROFILE):
        policies = _search_space(profile)["high_miss_no_admit_policy"]
        assert "bounded_stay_advance" in policies
        assert "repair_stop" in policies
        assert "legacy_advance_stay" not in policies


def test_strict_qpu_faithful_auto_euler_search_spaces_require_observable_guardrails() -> None:
    for profile in (
        realtime_optuna._STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE,
        realtime_optuna._STRICT_QPU_HH_RECOVERABILITY_PROFILE,
        realtime_optuna._STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE,
    ):
        space = _search_space(profile)
        assert space["integrator_policy"] == ["auto_euler_rk4"]
        for key in (
            "integrator_euler_site_span_max",
            "integrator_euler_primary_density_span_max",
            "integrator_euler_energy_span_max",
        ):
            assert key in space
            assert None not in space[key]
            assert all(float(value) > 0.0 for value in space[key])


def test_guardrail_only_profile_exposes_only_auto_euler_guardrails() -> None:
    profile = realtime_optuna._STRICT_QPU_FAITHFUL_APPEND_PRUNE_GUARDRAIL_ONLY_PROFILE
    space = _search_space(profile)
    assert set(space) == {
        "integrator_euler_site_span_max",
        "integrator_euler_primary_density_span_max",
        "integrator_euler_energy_span_max",
    }
    for choices in space.values():
        assert choices
        assert None not in choices
        assert all(float(value) > 0.0 for value in choices)


def test_guardrail_only_profile_freezes_class_lock_except_guardrails() -> None:
    profile = realtime_optuna._STRICT_QPU_FAITHFUL_APPEND_PRUNE_GUARDRAIL_ONLY_PROFILE
    proposed = TrialParams(
        prune_mode="off",
        append_no_harm_guard_enabled=False,
        miss_threshold=0.005,
        integrator_euler_site_span_max=0.02,
        integrator_euler_primary_density_span_max=0.005,
        integrator_euler_energy_span_max=0.005,
    )
    base_cfg = BaseRunConfig(
        artifact_json=Path("artifact.json"),
        study_profile=profile,
        tuning_class="bosonic",
    )

    frozen = realtime_optuna._apply_guardrail_only_class_lock_overrides(proposed, base_cfg)
    locked = realtime_optuna._locked_class_trial_params(base_cfg)

    assert frozen.prune_mode == locked.prune_mode == "schur_projected_shadow_v1"
    assert frozen.append_no_harm_guard_enabled is locked.append_no_harm_guard_enabled
    assert frozen.miss_threshold == locked.miss_threshold
    assert frozen.integrator_euler_site_span_max == pytest.approx(0.02)
    assert frozen.integrator_euler_primary_density_span_max == pytest.approx(0.005)
    assert frozen.integrator_euler_energy_span_max == pytest.approx(0.005)


def test_guardrail_only_profile_forwards_frozen_append_prune_tokens_exact_free() -> None:
    profile = realtime_optuna._STRICT_QPU_FAITHFUL_APPEND_PRUNE_GUARDRAIL_ONLY_PROFILE
    base_cfg = BaseRunConfig(
        artifact_json=Path("artifact.json"),
        study_profile=profile,
        tuning_class="bosonic",
    )
    params = realtime_optuna._apply_guardrail_only_class_lock_overrides(
        _baseline_trial_params(profile=profile),
        base_cfg,
    )

    tokens = _build_realtime_tokens(
        base_cfg=base_cfg,
        params=params,
        output_json=Path("out.json"),
        run_tag="guardrail_only",
    )

    assert tokens[tokens.index("--checkpoint-controller-mode") + 1] == "observable_v1"
    assert tokens[tokens.index("--checkpoint-controller-reference-mode") + 1] == "off"
    assert tokens[tokens.index("--checkpoint-controller-exact-input-mode") + 1] == "off"
    assert tokens[tokens.index("--checkpoint-controller-prune-mode") + 1] == "schur_projected_shadow_v1"
    assert "--checkpoint-controller-append-no-harm-guard-enabled" in tokens
    assert "exact_v1" not in tokens
    assert "benchmark_exact" not in tokens


def test_build_realtime_tokens_default_base_is_static_no_drive() -> None:
    base = BaseRunConfig(artifact_json=Path("artifact.json"))
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=TrialParams(),
        output_json=Path("out.json"),
        run_tag="trial_static_default",
    )
    assert "--enable-drive" not in tokens
    assert "--disable-drive" not in tokens
    assert "--drive-A" not in tokens
    assert tokens[tokens.index("--append-pool-family") + 1] == "match_replay"
    args = build_realtime_parser().parse_args(tokens)
    assert args.enable_drive is False
    assert args.disable_drive is False
    assert args.append_pool_family == "match_replay"


def test_optuna_parser_accepts_generic_and_legacy_strict_profiles() -> None:
    parser = realtime_optuna.build_parser()
    for profile in (
        realtime_optuna._STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE,
        realtime_optuna._STRICT_QPU_HH_RECOVERABILITY_PROFILE,
        realtime_optuna._STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE,
        realtime_optuna._STRICT_QPU_FAITHFUL_APPEND_PRUNE_GUARDRAIL_ONLY_PROFILE,
    ):
        args = parser.parse_args(["--artifact-json", "artifact.json", "--study-profile", profile])
        assert args.study_profile == profile


def test_optuna_parser_accepts_coarse_class_tuning_surface() -> None:
    parser = realtime_optuna.build_parser()
    args = parser.parse_args(
        [
            "--artifact-json",
            "artifact.json",
            "--tuning-class",
            "fermionic",
            "--class-settings-source",
            "paper_ii_class_tuning_defaults_v1",
            "--class-settings-output",
            "class_settings.json",
        ]
    )
    assert args.tuning_class == "fermionic"
    assert args.class_settings_source == "paper_ii_class_tuning_defaults_v1"
    assert args.class_settings_output == Path("class_settings.json")


@pytest.mark.parametrize(
    "profile",
    [
        realtime_optuna._STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE,
        realtime_optuna._STRICT_QPU_HH_RECOVERABILITY_PROFILE,
    ],
)
def test_strict_qpu_faithful_profile_tokens_are_qpu_faithful_and_exact_free(
    profile: str,
) -> None:
    tokens = _build_realtime_tokens(
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json"), study_profile=profile),
        params=_baseline_trial_params(profile=profile),
        output_json=Path("out.json"),
        run_tag="strict_trial",
    )

    assert "--checkpoint-controller-strict-qpu-faithful" in tokens
    assert "--checkpoint-controller-strict-qpu-hh" not in tokens
    assert tokens[tokens.index("--checkpoint-controller-mode") + 1] == "observable_v1"
    assert tokens[tokens.index("--checkpoint-controller-reference-mode") + 1] == "off"
    assert tokens[tokens.index("--checkpoint-controller-exact-input-mode") + 1] == "off"
    assert tokens[tokens.index("--checkpoint-controller-noise-mode") + 1] == "ideal"
    assert tokens[tokens.index("--checkpoint-controller-integrator-policy") + 1] == "auto_euler_rk4"
    assert tokens[tokens.index("--checkpoint-controller-prune-mode") + 1] == "off"
    assert "exact_v1" not in tokens
    assert "benchmark_exact" not in tokens
    assert not any("exact-forecast" in token for token in tokens)

    args = build_realtime_parser().parse_args(tokens)
    generic_args = build_generic_realtime_parser().parse_args(tokens)
    cfg = build_controller_config(args)
    assert args.checkpoint_controller_strict_qpu_faithful is True
    assert args.checkpoint_controller_strict_qpu_hh is False
    assert generic_args.checkpoint_controller_strict_qpu_faithful is True
    assert args.checkpoint_controller_noise_mode == "ideal"
    assert cfg.mode == "observable_v1"
    assert cfg.reference_mode == "off"
    assert cfg.integrator_policy == "auto_euler_rk4"
    assert cfg.integrator_euler_site_span_max == pytest.approx(
        realtime_optuna.STRICT_AUTO_EULER_SITE_SPAN_DEFAULT
    )
    assert cfg.integrator_euler_primary_density_span_max == pytest.approx(
        realtime_optuna.STRICT_AUTO_EULER_PRIMARY_DENSITY_SPAN_DEFAULT
    )
    assert cfg.integrator_euler_energy_span_max == pytest.approx(
        realtime_optuna.STRICT_AUTO_EULER_ENERGY_SPAN_DEFAULT
    )
    assert cfg.prune_mode == "off"
    # observable_v1 may keep the local observable-only no-harm guard enabled;
    # strictness is enforced by the decision contract, not by disabling
    # measurement-compatible guardrails.
    assert cfg.append_no_harm_guard_enabled is True


def test_strict_append_prune_profile_forwards_safe_append_prune_knobs_and_exact_free() -> None:
    profile = realtime_optuna._STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE
    params = _baseline_trial_params(profile=profile)
    tokens = _build_realtime_tokens(
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json"), study_profile=profile),
        params=params,
        output_json=Path("out.json"),
        run_tag="strict_append_prune_trial",
    )

    assert "--checkpoint-controller-strict-qpu-faithful" in tokens
    assert tokens[tokens.index("--checkpoint-controller-reference-mode") + 1] == "off"
    assert tokens[tokens.index("--checkpoint-controller-exact-input-mode") + 1] == "off"
    assert tokens[tokens.index("--checkpoint-controller-prune-mode") + 1] == "schur_projected_shadow_v1"
    assert "--checkpoint-controller-prune-projection-mode" in tokens
    assert "--checkpoint-controller-prune-shadow-enabled" in tokens
    assert "--checkpoint-controller-append-no-harm-guard-enabled" in tokens
    assert not any("exact_v1" in token for token in tokens)
    assert not any("benchmark_exact" in token for token in tokens)
    assert not any("exact-forecast" in token for token in tokens)

    args = build_realtime_parser().parse_args(tokens)
    cfg = build_controller_config(args)
    assert cfg.prune_mode == "schur_projected_shadow_v1"
    assert cfg.prune_projection_mode == "state_tangent_ls_v1"
    assert cfg.append_no_harm_guard_enabled is True


def test_strict_append_prune_objective_uses_opportunity_gated_shortfalls(tmp_path: Path) -> None:
    profile = realtime_optuna._STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE
    payload = _strict_synthetic_payload(family="hubbard", strict_qpu_hh=False)
    payload["summary"].update({"high_miss_count": 1, "append_count": 0, "prune_count": 0})
    payload["ledger"][1]["proposed_action_kind"] = "append_candidate"
    payload["ledger"][2]["prune_candidates"] = [{"coordinate": 0}]
    output_json = tmp_path / "strict_append_prune.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            study_profile=profile,
            t_final=2.0,
            num_times=3,
        ),
    )
    weights = ObjectiveWeights(append_count=0.08, prune_count=0.05)
    improved = dict(metrics)
    improved["append_count"] = 1
    improved["prune_count"] = 1

    assert metrics["append_opportunity_count"] == 1
    assert metrics["prune_opportunity_count"] == 1
    assert _objective_value(metrics, weights) > _objective_value(improved, weights)


def test_strict_qpu_faithful_profile_with_drive_forwards_drive_args_and_exact_free() -> None:
    profile = realtime_optuna._STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE
    tokens = _build_realtime_tokens(
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            study_profile=profile,
            enable_drive=True,
            drive_A=0.55,
            drive_omega=1.25,
            drive_tbar=4.0,
            drive_phi=0.125,
            drive_pattern="custom",
            drive_custom_weights="1.0,-1.0",
            drive_include_identity=True,
            drive_time_sampling="midpoint",
            drive_t0=4.0,
        ),
        params=_baseline_trial_params(profile=profile),
        output_json=Path("out.json"),
        run_tag="strict_driven_trial",
    )

    assert "--checkpoint-controller-strict-qpu-faithful" in tokens
    assert "--checkpoint-controller-strict-qpu-hh" not in tokens
    assert tokens[tokens.index("--checkpoint-controller-mode") + 1] == "observable_v1"
    assert tokens[tokens.index("--checkpoint-controller-reference-mode") + 1] == "off"
    assert tokens[tokens.index("--checkpoint-controller-exact-input-mode") + 1] == "off"
    assert tokens[tokens.index("--checkpoint-controller-noise-mode") + 1] == "ideal"
    assert tokens[tokens.index("--checkpoint-controller-integrator-policy") + 1] == "auto_euler_rk4"
    assert tokens[tokens.index("--checkpoint-controller-prune-mode") + 1] == "off"
    assert "exact_v1" not in tokens
    assert "benchmark_exact" not in tokens
    assert not any("exact-forecast" in token for token in tokens)

    assert "--enable-drive" in tokens
    assert "--disable-drive" not in tokens
    assert tokens[tokens.index("--drive-A") + 1] == "0.55"
    assert tokens[tokens.index("--drive-omega") + 1] == "1.25"
    assert tokens[tokens.index("--drive-tbar") + 1] == "4.0"
    assert tokens[tokens.index("--drive-phi") + 1] == "0.125"
    assert tokens[tokens.index("--drive-pattern") + 1] == "custom"
    assert tokens[tokens.index("--drive-custom-weights") + 1] == "1.0,-1.0"
    assert "--drive-include-identity" in tokens
    assert tokens[tokens.index("--drive-time-sampling") + 1] == "midpoint"
    assert tokens[tokens.index("--drive-t0") + 1] == "4.0"

    args = build_realtime_parser().parse_args(tokens)
    generic_args = build_generic_realtime_parser().parse_args(tokens)
    cfg = build_controller_config(args)
    assert args.enable_drive is True
    assert args.disable_drive is False
    assert args.drive_A == pytest.approx(0.55)
    assert args.drive_omega == pytest.approx(1.25)
    assert args.drive_tbar == pytest.approx(4.0)
    assert args.drive_phi == pytest.approx(0.125)
    assert args.drive_pattern == "custom"
    assert args.drive_custom_weights == "1.0,-1.0"
    assert args.drive_include_identity is True
    assert args.drive_time_sampling == "midpoint"
    assert args.drive_t0 == pytest.approx(4.0)
    assert args.checkpoint_controller_strict_qpu_faithful is True
    assert args.checkpoint_controller_strict_qpu_hh is False
    assert generic_args.checkpoint_controller_strict_qpu_faithful is True
    assert cfg.mode == "observable_v1"
    assert cfg.reference_mode == "off"
    assert cfg.integrator_policy == "auto_euler_rk4"
    assert cfg.prune_mode == "off"


def test_build_realtime_tokens_explicit_append_pool_family_round_trip() -> None:
    base = BaseRunConfig(artifact_json=Path("artifact.json"), append_pool_family="full_meta")
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=TrialParams(),
        output_json=Path("out.json"),
        run_tag="full_meta_append",
    )
    args = build_realtime_parser().parse_args(tokens)
    assert tokens[tokens.index("--append-pool-family") + 1] == "full_meta"
    assert args.append_pool_family == "full_meta"


def test_build_realtime_tokens_enable_drive_emits_neutral_defaults() -> None:
    base = BaseRunConfig(artifact_json=Path("artifact.json"), enable_drive=True)
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=TrialParams(),
        output_json=Path("out.json"),
        run_tag="trial_drive_default",
    )
    assert "--enable-drive" in tokens
    assert tokens[tokens.index("--drive-A") + 1] == "0.0"
    assert tokens[tokens.index("--drive-omega") + 1] == "1.0"
    assert tokens[tokens.index("--drive-tbar") + 1] == "1.0"
    args = build_realtime_parser().parse_args(tokens)
    assert args.enable_drive is True
    assert args.drive_A == pytest.approx(0.0)
    assert args.drive_omega == pytest.approx(1.0)
    assert args.drive_tbar == pytest.approx(1.0)


def test_build_realtime_tokens_rejects_conflicting_drive_controls() -> None:
    base = BaseRunConfig(
        artifact_json=Path("artifact.json"),
        enable_drive=True,
        disable_drive=True,
    )
    with pytest.raises(ValueError):
        _build_realtime_tokens(
            base_cfg=base,
            params=TrialParams(),
            output_json=Path("out.json"),
            run_tag="trial_conflict",
        )


def test_optuna_parser_rejects_enable_and_disable_drive() -> None:
    with pytest.raises(SystemExit):
        realtime_optuna.build_parser().parse_args(
            ["--artifact-json", "artifact.json", "--enable-drive", "--disable-drive"]
        )


def test_optuna_parser_defaults_to_json_only_spectra_and_accepts_pdf_controls() -> None:
    parser = realtime_optuna.build_parser()
    default_args = parser.parse_args(["--artifact-json", "artifact.json"])
    assert default_args.skip_spectra_pdf is True
    assert realtime_optuna._build_base_run_config(default_args).skip_spectra_pdf is True

    skip_args = parser.parse_args(["--artifact-json", "artifact.json", "--skip-spectra-pdf"])
    assert skip_args.skip_spectra_pdf is True

    opt_in_args = parser.parse_args(["--artifact-json", "artifact.json", "--with-spectra-pdf"])
    assert opt_in_args.skip_spectra_pdf is False
    assert realtime_optuna._build_base_run_config(opt_in_args).skip_spectra_pdf is False


def test_build_realtime_tokens_keeps_isolated_override_surface() -> None:
    base = BaseRunConfig(artifact_json=Path("artifact.json"), disable_drive=True)
    params = TrialParams(
        horizon_mode="lead3",
        step_scale_mode="wide",
        blend_weight_mode="tight",
        gain_scale_mode="mild",
        baseline_step_refine_rounds=1,
        miss_threshold=0.03,
        gain_ratio_threshold=0.01,
        trust_radius=1.25,
        signed_energy_lead_limit=2.0,
        primary_density_weight=4.0,
        site_weight=2.0,
        energy_weight=0.25,
        density_slope_weight=2.0,
    )
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=params,
        output_json=Path("out.json"),
        run_tag="trial_0000",
        progress_json=Path("progress.json"),
        partial_payload_json=Path("partial.json"),
    )
    assert "--artifact-json" in tokens
    assert "--disable-drive" in tokens
    assert "--checkpoint-controller-exact-forecast-horizon-steps" in tokens
    assert tokens[tokens.index("--checkpoint-controller-exact-forecast-horizon-steps") + 1] == "3"
    assert tokens[tokens.index("--checkpoint-controller-exact-forecast-tangent-secant-trust-radius") + 1] == "1.25"
    assert tokens[tokens.index("--checkpoint-controller-exact-forecast-tracking-primary-density-error-weight") + 1] == "4.0"
    assert tokens[tokens.index("--checkpoint-controller-integrator-policy") + 1] == "auto_euler_rk4"
    assert tokens[tokens.index("--checkpoint-controller-integrator-euler-min-time-fraction") + 1] == "0.35"
    assert tokens[tokens.index("--checkpoint-controller-integrator-euler-observable-window") + 1] == "16"
    assert "--checkpoint-controller-integrator-euler-site-span-max" not in tokens
    assert "--checkpoint-controller-integrator-euler-primary-density-span-max" not in tokens
    assert "--checkpoint-controller-integrator-euler-energy-span-max" not in tokens
    assert tokens[tokens.index("--progress-json") + 1] == "progress.json"
    assert tokens[tokens.index("--partial-payload-json") + 1] == "partial.json"


def test_build_realtime_tokens_round_trip_through_current_realtime_parser() -> None:
    base = BaseRunConfig(artifact_json=Path("artifact.json"), disable_drive=True)
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=TrialParams(blend_weight_mode="default", gain_scale_mode="mild"),
        output_json=Path("out.json"),
        run_tag="trial_0001",
        progress_json=Path("progress.json"),
        partial_payload_json=Path("partial.json"),
    )
    args = build_realtime_parser().parse_args(tokens)
    assert args.checkpoint_controller_exact_forecast_baseline_blend_weights.startswith("-0.25")
    assert args.checkpoint_controller_exact_forecast_baseline_gain_scales == "0.75,1.0,1.25"
    assert args.progress_json == "progress.json"
    assert args.partial_payload_json == "partial.json"


def test_build_realtime_tokens_round_trip_preserves_blend_off() -> None:
    base = BaseRunConfig(artifact_json=Path("artifact.json"), disable_drive=True)
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=TrialParams(blend_weight_mode="off", gain_scale_mode="off"),
        output_json=Path("out.json"),
        run_tag="trial_0002",
    )
    args = build_realtime_parser().parse_args(tokens)
    assert args.checkpoint_controller_exact_forecast_baseline_blend_weights == ""


def test_build_controller_config_maps_expanded_knobs_from_parser_args() -> None:
    args = build_realtime_parser().parse_args(
        [
            "--artifact-json",
            "artifact.json",
            "--output-json",
            "out.json",
            "--checkpoint-controller-oracle-selection-policy",
            "measured_topk_oracle_energy",
            "--no-checkpoint-controller-exact-forecast-include-tangent-secant-proposal",
            "--checkpoint-controller-exact-forecast-baseline-proposal-mode",
            "anticipatory_drive_basis_v1",
            "--checkpoint-controller-exact-forecast-primary-density-target-mode",
            "pair_difference",
            "--checkpoint-controller-append-margin-abs",
            "1e-5",
            "--checkpoint-controller-high-miss-no-admit-policy",
            "repair_stop",
            "--checkpoint-controller-miss-abs-threshold",
            "0.125",
            "--checkpoint-controller-miss-persistence-window",
            "3",
            "--checkpoint-controller-miss-persistence-count",
            "2",
            "--checkpoint-controller-integrator-policy",
            "rk4",
            "--checkpoint-controller-integrator-columnarity-threshold",
            "0.9",
            "--checkpoint-controller-integrator-curvature-threshold",
            "0.2",
            "--checkpoint-controller-integrator-euler-fs-error-threshold",
            "0.004",
            "--checkpoint-controller-integrator-condition-max",
            "1234.5",
            "--checkpoint-controller-integrator-euler-min-time-fraction",
            "0.4",
            "--checkpoint-controller-integrator-euler-observable-window",
            "8",
            "--checkpoint-controller-integrator-euler-site-span-max",
            "0.006",
            "--checkpoint-controller-integrator-euler-primary-density-span-max",
            "0.007",
            "--checkpoint-controller-integrator-euler-energy-span-max",
            "0.008",
            "--checkpoint-controller-shortlist-size",
            "6",
            "--checkpoint-controller-shortlist-fraction",
            "0.25",
            "--checkpoint-controller-active-window-size",
            "4",
            "--checkpoint-controller-max-probe-positions",
            "6",
            "--checkpoint-controller-regularization-lambda",
            "1e-6",
            "--checkpoint-controller-candidate-regularization-lambda",
            "1e-10",
            "--checkpoint-controller-pinv-rcond",
            "1e-8",
            "--checkpoint-controller-compile-penalty-weight",
            "0.0",
            "--checkpoint-controller-measurement-penalty-weight",
            "0.05",
            "--checkpoint-controller-directional-penalty-weight",
            "0.03",
            "--checkpoint-controller-confirm-score-mode",
            "exact_gain_ratio",
            "--checkpoint-controller-confirm-compress-fraction",
            "0.25",
            "--checkpoint-controller-confirm-compress-min-modes",
            "2",
            "--checkpoint-controller-confirm-compress-max-modes",
            "12",
        ]
    )
    cfg = build_controller_config(args)
    assert cfg.oracle_selection_policy == "measured_topk_oracle_energy"
    assert cfg.exact_forecast_include_tangent_secant_proposal is False
    assert cfg.exact_forecast_baseline_proposal_mode == "anticipatory_drive_basis_v1"
    assert cfg.exact_forecast_primary_density_target_mode == "pair_difference"
    assert cfg.append_margin_abs == pytest.approx(1.0e-5)
    assert cfg.high_miss_no_admit_policy == "repair_stop"
    assert cfg.miss_abs_threshold == pytest.approx(0.125)
    assert cfg.miss_persistence_window == 3
    assert cfg.miss_persistence_count == 2
    assert cfg.integrator_policy == "rk4"
    assert cfg.integrator_columnarity_threshold == pytest.approx(0.9)
    assert cfg.integrator_curvature_threshold == pytest.approx(0.2)
    assert cfg.integrator_euler_fs_error_threshold == pytest.approx(0.004)
    assert cfg.integrator_condition_max == pytest.approx(1234.5)
    assert cfg.integrator_euler_min_time_fraction == pytest.approx(0.4)
    assert cfg.integrator_euler_observable_window == 8
    assert cfg.integrator_euler_site_span_max == pytest.approx(0.006)
    assert cfg.integrator_euler_primary_density_span_max == pytest.approx(0.007)
    assert cfg.integrator_euler_energy_span_max == pytest.approx(0.008)
    assert cfg.shortlist_size == 6
    assert cfg.shortlist_fraction == pytest.approx(0.25)
    assert cfg.active_window_size == 4
    assert cfg.max_probe_positions == 6
    assert cfg.regularization_lambda == pytest.approx(1.0e-6)
    assert cfg.candidate_regularization_lambda == pytest.approx(1.0e-10)
    assert cfg.pinv_rcond == pytest.approx(1.0e-8)
    assert cfg.compile_penalty_weight == pytest.approx(0.0)
    assert cfg.measurement_penalty_weight == pytest.approx(0.05)
    assert cfg.directional_penalty_weight == pytest.approx(0.03)
    assert cfg.confirm_score_mode == "exact_gain_ratio"
    assert cfg.confirm_compress_fraction == pytest.approx(0.25)
    assert cfg.confirm_compress_min_modes == 2
    assert cfg.confirm_compress_max_modes == 12


def test_trial_metrics_from_payload_computes_density_and_spectral_metrics(tmp_path: Path) -> None:
    payload = _synthetic_payload()
    output_json = tmp_path / "result.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    analysis, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            pair="auto",
            spectra_detrend="linear",
            spectra_window="hann",
        ),
    )
    assert analysis["metadata"]["pair_difference"] == [0, 1]
    manifest = analysis["metadata"]["parameter_manifest"]
    assert manifest["compiled_count_2q"] == 42
    assert manifest["compiled_depth"] == 99
    assert manifest["compiled_size"] == 123
    assert manifest["compile_backend"] == "FakeMarrakesh"
    assert manifest["transpile_seed"] == 7
    assert manifest["transpile_optimization_level"] == 2
    assert metrics["compiled_count_2q"] == 42
    assert metrics["compiled_depth"] == 99
    assert metrics["compile_backend"] == "FakeMarrakesh"
    assert metrics["append_count"] == 1
    assert metrics["full_horizon_gate_passed"] is False
    assert metrics["full_horizon_gate_reason"] == "final_time_short"
    assert metrics["high_miss_no_admit_soft_fallback_count"] == 0
    assert metrics["high_miss_no_admit_soft_fallback_fraction"] == pytest.approx(0.0)
    assert metrics["ordinary_stay_count"] == 0
    assert metrics["final_runtime_parameter_count"] == 4
    assert metrics["min_fidelity_exact"] == pytest.approx(0.9)
    assert metrics["pair_span_exact"] == pytest.approx(0.4)
    assert metrics["pair_mae"] == pytest.approx(0.2)
    assert metrics["pair_mae_over_exact_span"] == pytest.approx(0.5)
    assert metrics["mean_abs_site_occupations_error"] == pytest.approx((0.0 + 0.1 + 0.2) / 3.0)
    assert metrics["mean_abs_energy_total_error"] == pytest.approx((0.0 + 0.1 + 0.15) / 3.0)
    assert metrics["final_abs_energy_total_error"] == pytest.approx(0.15)
    assert metrics["mean_total_occupation_abs_error"] == pytest.approx(0.0)
    assert metrics["mean_tracking_d_curvature_abs_error_mean"] == pytest.approx((0.0 + 0.2 + 0.4) / 3.0)
    assert metrics["mean_tracking_d_excursion_under_response_mean"] == pytest.approx((0.0 + 0.3 + 0.6) / 3.0)
    assert metrics["mean_tracking_d_excursion_over_response_mean"] == pytest.approx((0.0 + 0.1 + 0.2) / 3.0)
    assert metrics["dominant_peak_abs_omega_error"] >= 0.0
    assert "drive_line_ratio_defect" in metrics


def test_trial_metrics_from_payload_short_early_stop_is_penalized_not_failed(tmp_path: Path) -> None:
    payload = _short_early_stop_payload()
    output_json = tmp_path / "short_result.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    analysis, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json"), pair="auto"),
    )

    assert analysis["metadata"]["analysis_status"] == "spectra_unavailable"
    assert metrics["analysis_status"] == "spectra_unavailable"
    assert "Need at least two time samples" in str(metrics["analysis_error"])
    assert metrics["trajectory_row_count"] == 1
    assert metrics["summary_status"] == "repair_stop"
    assert metrics["early_stop_reason"] == "high_miss_no_admit_repair_stop"
    assert metrics["mean_abs_energy_total_error"] == pytest.approx(0.25)
    assert metrics["final_abs_energy_total_error"] == pytest.approx(0.25)
    assert metrics["mean_abs_site_occupations_error"] == pytest.approx(0.0)
    assert metrics["min_fidelity_exact"] == pytest.approx(0.8)
    assert math.isnan(float(metrics["dominant_peak_abs_omega_error"]))
    assert math.isnan(float(metrics["drive_line_ratio_defect"]))
    reasons = _invalid_reasons(metrics)
    assert metrics["trajectory_reached_final_time"] is False
    assert metrics["trajectory_reached_expected_rows"] is False
    assert metrics["full_horizon_gate_passed"] is False
    assert metrics["full_horizon_gate_reason"] == "early_stop:high_miss_no_admit_repair_stop"
    assert "missing_or_nonfinite:pair_mae_over_exact_span" not in reasons
    assert "missing_or_nonfinite:epsilon_osc_pair" not in reasons
    assert "missing_or_nonfinite:dominant_peak_abs_omega_error" not in reasons
    assert "trajectory_early_stop:high_miss_no_admit_repair_stop" in reasons
    assert "trajectory_incomplete:final_time" in reasons
    assert "trajectory_incomplete:row_count" in reasons
    assert "full_horizon_gate:early_stop:high_miss_no_admit_repair_stop" in reasons


def test_trial_metrics_from_payload_propagates_controller_stable_early_stop_reason(tmp_path: Path) -> None:
    payload = _synthetic_payload()
    reason = "progress_observables_stable:site_span<=0.0025,checkpoint>=160"
    payload["summary"].update(
        {
            "status": "stopped_early",
            "early_stop_reason": reason,
            "full_horizon_gate_passed": True,
            "full_horizon_successful_early_stop": True,
            "full_horizon_completion_kind": "stable_early_stop",
        }
    )
    payload["trajectory"] = list(payload["trajectory"])[:2]
    output_json = tmp_path / "stable_stop_result.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _analysis, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            study_profile="append_prune_noharm_l2_v1",
            pair="auto",
            t_final=8.0,
            num_times=321,
        ),
    )

    assert metrics["full_horizon_gate_passed"] is True
    assert metrics["full_horizon_successful_early_stop"] is True
    assert metrics["full_horizon_completion_kind"] == "stable_early_stop"
    assert metrics["full_horizon_early_stop_reason"] == reason
    reasons = _invalid_reasons(metrics)
    assert "trajectory_incomplete:final_time" not in reasons
    assert "trajectory_incomplete:row_count" not in reasons


def test_generic_exact_v1_primary_observable_metrics_do_not_require_pair_or_spectrum(
    tmp_path: Path,
) -> None:
    payload = {
        "run_tag": "spin_boson_primary_only",
        "summary": {
            "status": "completed",
            "append_count": 0,
            "prune_count": 0,
            "final_runtime_parameter_count": 2,
            "max_abs_energy_total_error": 0.02,
            "max_abs_site_occupations_error": 0.05,
        },
        "trajectory": [
            {
                "time": 0.0,
                "physical_time": 0.0,
                "primary_density_mode": "imbalance",
                "primary_density": 0.10,
                "primary_density_exact": 0.10,
                "site_occupations": [0.50],
                "site_occupations_exact": [0.50],
                "energy_total": 0.0,
                "energy_total_exact": 0.0,
                "abs_energy_total_error": 0.0,
                "fidelity_exact": 1.0,
                "site_occupations_abs_error_max": 0.0,
                "runtime_parameter_count": 2,
            },
            {
                "time": 1.0,
                "physical_time": 1.0,
                "primary_density_mode": "imbalance",
                "primary_density": 0.20,
                "primary_density_exact": 0.40,
                "site_occupations": [0.55],
                "site_occupations_exact": [0.50],
                "energy_total": 0.02,
                "energy_total_exact": 0.0,
                "abs_energy_total_error": 0.02,
                "fidelity_exact": 0.98,
                "site_occupations_abs_error_max": 0.05,
                "runtime_parameter_count": 2,
            },
        ],
    }
    output_json = tmp_path / "spin_boson_primary_only.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    analysis, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("spin_boson_artifact.json"),
            study_profile="generic_l2_exact_v1",
            pair="auto",
            t_final=1.0,
            num_times=2,
        ),
    )
    reasons = _invalid_reasons(metrics)
    objective = _objective_value(metrics, ObjectiveWeights())

    assert metrics["generic_exact_v1_family_objective"] is True
    assert metrics["primary_observable_name"] == "primary_density"
    assert metrics["primary_observable_source"] == "trajectory.primary_density"
    assert metrics["primary_observable_num_samples"] == 2
    assert metrics["primary_observable_mae_over_exact_span"] == pytest.approx(1.0 / 3.0)
    assert metrics["primary_observable_spectrum_available"] is False
    assert analysis["metadata"]["primary_observable_source"] == "trajectory.primary_density"
    assert "missing_or_nonfinite:pair_mae_over_exact_span" not in reasons
    assert "missing_or_nonfinite:epsilon_osc_pair" not in reasons
    assert "missing_or_nonfinite:dominant_peak_abs_omega_error" not in reasons
    assert reasons == []
    assert math.isfinite(float(objective))
    assert float(objective) < realtime_optuna._LARGE_OBJECTIVE


def test_generic_exact_v1_primary_observable_falls_back_to_staggered_site_data(
    tmp_path: Path,
) -> None:
    payload = {
        "run_tag": "ttprime_staggered_fallback",
        "summary": {
            "status": "completed",
            "append_count": 0,
            "prune_count": 0,
            "final_runtime_parameter_count": 4,
            "max_abs_energy_total_error": 0.03,
            "max_abs_site_occupations_error": 0.1,
        },
        "trajectory": [
            {
                "time": 0.0,
                "physical_time": 0.0,
                "primary_density_mode": "staggered",
                "site_occupations": [1.1, 0.0, 0.0],
                "site_occupations_exact": [1.0, 0.0, 0.0],
                "energy_total": 0.0,
                "energy_total_exact": 0.0,
                "abs_energy_total_error": 0.0,
                "fidelity_exact": 1.0,
                "site_occupations_abs_error_max": 0.1,
                "runtime_parameter_count": 4,
            },
            {
                "time": 1.0,
                "physical_time": 1.0,
                "primary_density_mode": "staggered",
                "site_occupations": [0.0, 0.8, 0.0],
                "site_occupations_exact": [0.0, 1.0, 0.0],
                "energy_total": 0.03,
                "energy_total_exact": 0.0,
                "abs_energy_total_error": 0.03,
                "fidelity_exact": 0.97,
                "site_occupations_abs_error_max": 0.2,
                "runtime_parameter_count": 4,
            },
        ],
    }
    output_json = tmp_path / "ttprime_staggered_fallback.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("ttprime_hubbard_artifact.json"),
            study_profile="generic_l2_exact_v1",
            pair="auto",
            t_final=1.0,
            num_times=2,
        ),
    )
    reasons = _invalid_reasons(metrics)

    assert metrics["primary_observable_name"] == "staggered"
    assert metrics["primary_observable_source"] == "site_occupations.staggered_fallback"
    assert math.isfinite(float(metrics["primary_observable_mae_over_exact_span"]))
    assert "missing_or_nonfinite:pair_mae_over_exact_span" not in reasons
    assert "missing_or_nonfinite:epsilon_osc_pair" not in reasons
    assert "missing_or_nonfinite:dominant_peak_abs_omega_error" not in reasons
    assert reasons == []
    assert _objective_value(metrics, ObjectiveWeights()) < realtime_optuna._LARGE_OBJECTIVE


def test_trial_metrics_filters_repair_events_for_expected_rows(tmp_path: Path) -> None:
    payload = _synthetic_payload()
    repair_row = dict(payload["trajectory"][0])
    repair_row.update(
        {
            "action_kind": "repair_miss",
            "trajectory_sample_kind": "repair_event",
            "advances_time": False,
            "repair_retry_next": True,
        }
    )
    payload["trajectory"].insert(1, repair_row)
    output_json = tmp_path / "retry_success_result.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    analysis, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json"), pair="auto", t_final=2.0, num_times=3),
    )
    reasons = _invalid_reasons(metrics)

    assert analysis["metadata"]["repair_event_row_count"] == 1
    assert metrics["trajectory_row_count"] == 4
    assert metrics["raw_trajectory_row_count"] == 4
    assert metrics["repair_event_row_count"] == 1
    assert metrics["trajectory_state_sample_count"] == 3
    assert metrics["trajectory_reached_expected_rows"] is True
    assert metrics["trajectory_reached_final_time"] is True
    assert metrics["full_horizon_gate_passed"] is True
    assert metrics["full_horizon_gate_reason"] == "passed"
    assert "trajectory_incomplete:row_count" not in reasons
    assert "trajectory_incomplete:final_time" not in reasons
    assert not any(reason.startswith("full_horizon_gate:") for reason in reasons)


def test_strict_trial_metrics_skip_exact_analysis_and_pdf_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = realtime_optuna._STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE

    def _raise_exact_analysis(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("strict profile called exact/spectra analysis")

    def _fake_run(*, args: object, params: TrialParams) -> dict[str, object]:
        del params
        payload = _strict_synthetic_payload(family="hubbard", strict_qpu_hh=False)
        output_json = Path(str(getattr(args, "output_json")))
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    monkeypatch.setattr(realtime_optuna, "load_trajectory_payload", _raise_exact_analysis)
    monkeypatch.setattr(realtime_optuna, "analyze_payload", _raise_exact_analysis)
    monkeypatch.setattr(realtime_optuna, "render_spectrum_pdf", _raise_exact_analysis)
    monkeypatch.setattr(
        realtime_optuna,
        "_run_realtime_from_args_with_optuna_overrides",
        _fake_run,
    )

    obs = _evaluate_trial(
        trial_number=0,
        params=_baseline_trial_params(profile=profile),
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            study_profile=profile,
            t_final=2.0,
            num_times=3,
            skip_spectra_pdf=False,
        ),
        objective_weights=ObjectiveWeights(),
        validity_gates=ValidityGates(),
        output_dir=tmp_path,
    )

    assert obs.status == "completed"
    assert obs.spectra_json is not None
    assert obs.spectra_pdf is None
    assert obs.metrics["strict_qpu_faithful"] is True
    assert obs.metrics["strict_qpu_hh"] is False
    assert obs.metrics["strict_qpu_family"] == "hubbard"
    analysis = json.loads(Path(obs.spectra_json).read_text(encoding="utf-8"))
    assert analysis["metadata"]["analysis_status"] == "strict_qpu_faithful_exact_analysis_skipped"
    assert analysis["metadata"]["recoverability_reference"]["controller_decision_use"] == "forbidden"
    assert not (tmp_path / "trials" / "trial_0000" / "spectra.pdf").exists()


@pytest.mark.parametrize(
    ("family", "strict_qpu_hh"),
    [("hh", True), ("hubbard", False)],
)
def test_strict_objective_is_invariant_to_exact_only_payload_fields(
    tmp_path: Path,
    family: str,
    strict_qpu_hh: bool,
) -> None:
    profile = realtime_optuna._STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE
    payload = _strict_synthetic_payload(family=family, strict_qpu_hh=strict_qpu_hh)
    output_json = tmp_path / f"strict_{family}_result.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            study_profile=profile,
            t_final=2.0,
            num_times=3,
        ),
    )

    mutated = json.loads(json.dumps(payload))
    mutated["summary"]["final_abs_energy_total_error"] = 999999.0
    mutated["summary"]["final_fidelity_exact"] = -999999.0
    mutated["diagnostic_reference"] = {
        "feeds_controller_decisions": False,
        "energy_total_exact": [1.0e9],
    }
    for idx, row in enumerate(mutated["trajectory"]):
        row["energy_total_exact"] = -1.0e6 * (idx + 1)
        row["abs_energy_total_error"] = 1.0e6 * (idx + 1)
        row["fidelity_exact"] = -100.0 * (idx + 1)
        row["site_occupations_exact"] = [1.0e6, -1.0e6]
        row["site_occupations_abs_error_max"] = 1.0e6
    mutated["ledger"] = [dict(row) for row in mutated["trajectory"]]
    mutated_output_json = tmp_path / f"strict_{family}_result_mutated.json"
    mutated_output_json.write_text(json.dumps(mutated, indent=2), encoding="utf-8")
    _, mutated_metrics = _trial_metrics_from_payload(
        payload=mutated,
        output_json=mutated_output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            study_profile=profile,
            t_final=2.0,
            num_times=3,
        ),
    )

    weights = ObjectiveWeights()
    assert metrics["strict_qpu_faithful"] is True
    assert metrics["strict_qpu_hh"] is strict_qpu_hh
    assert metrics["strict_qpu_family"] == family
    assert _objective_value(mutated_metrics, weights) == pytest.approx(
        _objective_value(metrics, weights)
    )
    assert _invalid_reasons(mutated_metrics) == _invalid_reasons(metrics)
    assert mutated_metrics["mean_rho_miss"] == pytest.approx(metrics["mean_rho_miss"])
    assert mutated_metrics["strict_measured_degradation_score"] == pytest.approx(
        metrics["strict_measured_degradation_score"]
    )
    assert mutated_metrics["strict_post_first_euler_euler_count"] == metrics["strict_post_first_euler_euler_count"]
    assert mutated_metrics["strict_tail_euler_count"] == metrics["strict_tail_euler_count"]


def test_strict_metrics_emit_exact_free_tail_observable_telemetry(tmp_path: Path) -> None:
    payload = _strict_synthetic_payload(family="hh", strict_qpu_hh=True)
    trajectory = []
    for idx, row in enumerate(payload["trajectory"]):
        enriched = dict(row)
        enriched.update(
            {
                "time": float(idx),
                "physical_time": float(idx),
                "integrator_policy": "auto_euler_rk4",
                "integrator_used": "rk4" if idx == 0 else ("euler" if idx == 1 else "none"),
                "energy_total": float(idx) * 0.25,
                "primary_density": float(idx) * 0.5,
                "site_occupations": [1.0 + 0.1 * float(idx), 1.0 - 0.1 * float(idx)],
                # Exact fields are intentionally inconsistent and must not affect strict tail telemetry.
                "energy_total_exact": -999.0 * float(idx + 1),
                "site_occupations_exact": [999.0, -999.0],
            }
        )
        trajectory.append(enriched)
    payload["trajectory"] = trajectory
    payload["ledger"] = [dict(row) for row in trajectory]
    output_json = tmp_path / "strict_tail_metrics.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            study_profile=realtime_optuna._STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE,
            t_final=2.0,
            num_times=3,
        ),
    )

    assert metrics["strict_observable_tail_metrics_schema"] == "strict_observable_tail_metrics_v1"
    assert metrics["strict_observable_tail_metrics_exact_free"] is True
    assert metrics["strict_first_euler_time"] == pytest.approx(1.0)
    assert metrics["strict_post_first_euler_sample_count"] == 2
    assert metrics["strict_post_first_euler_euler_count"] == 1
    assert metrics["strict_post_first_euler_site0_range"] == pytest.approx(0.1)
    assert metrics["strict_post_first_euler_energy_range"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("mutate", "expected_contract_reason"),
    [
        (
            lambda payload: payload["summary"].update({"exact_decision_checkpoints": 1}),
            "summary.exact_decision_checkpoints=1",
        ),
        (
            lambda payload: payload["summary"].update({"reference_enabled": True}),
            "summary.reference_enabled=true",
        ),
        (
            lambda payload: payload["reference"].update({"reference_enabled": True}),
            "reference.reference_enabled=true",
        ),
        (
            lambda payload: payload["summary"].update({"reference_mode": "benchmark_exact"}),
            "summary.reference_mode=benchmark_exact",
        ),
        (
            lambda payload: payload["reference"].update({"reference_mode": "benchmark_exact"}),
            "reference.reference_mode=benchmark_exact",
        ),
        (
            lambda payload: payload["summary"].update({"exact_audit_helper_active": True}),
            "summary.exact_audit_helper_active=active",
        ),
        (
            lambda payload: payload["ledger"][1].update({"decision_backend": "exact"}),
            "row[1].decision_backend=exact",
        ),
        (
            lambda payload: payload["ledger"][1].update({"exact_forecast_error": "used"}),
            "row[1].exact_forecast_error=present",
        ),
        (
            lambda payload: payload["ledger"][1].update(
                {"decision_override_reason": "exact_forecast_dual_metric_regression"}
            ),
            "row[1].decision_override_reason=exact_forecast_dual_metric_regression",
        ),
        (
            lambda payload: payload["ledger"][1].update(
                {"exact_v1_selection_reason": "exact_v1_candidate_selected"}
            ),
            "row[1].exact_v1_selection_reason=present",
        ),
    ],
)
def test_strict_metrics_contract_invalidates_exact_decision_leaks(
    tmp_path: Path,
    mutate,
    expected_contract_reason: str,
) -> None:
    profile = realtime_optuna._STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE
    payload = _strict_synthetic_payload(family="hubbard", strict_qpu_hh=False)
    mutate(payload)
    output_json = tmp_path / "strict_contract_leak.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            study_profile=profile,
            t_final=2.0,
            num_times=3,
        ),
    )
    reasons = _invalid_reasons(metrics)

    assert metrics["strict_qpu_faithful"] is True
    assert metrics["strict_qpu_hh"] is False
    assert metrics["strict_decision_contract_passed"] is False
    assert metrics["qpu_faithful_decisions_passed"] is False
    assert expected_contract_reason in metrics["strict_decision_contract_violations"]
    assert f"strict_purity_contract:{expected_contract_reason}" in reasons


def test_strict_metrics_do_not_emit_exact_objective_feedback_fields(tmp_path: Path) -> None:
    profile = realtime_optuna._STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE
    payload = _strict_synthetic_payload(family="hubbard", strict_qpu_hh=False)
    output_json = tmp_path / "strict_no_exact_feedback.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            study_profile=profile,
            t_final=2.0,
            num_times=3,
        ),
    )

    exact_feedback_keys = {
        "pair_mae_over_exact_span",
        "mean_abs_energy_total_error",
        "max_abs_energy_total_error",
        "mean_abs_site_occupations_error",
        "min_fidelity_exact",
        "dominant_peak_abs_omega_error",
        "epsilon_osc_pair",
    }
    assert metrics["qpu_faithful_decisions_passed"] is True
    assert exact_feedback_keys.isdisjoint(metrics.keys())


def test_strict_metrics_allows_inactive_exact_forecast_veto_counter(tmp_path: Path) -> None:
    profile = realtime_optuna._STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE
    payload = _strict_synthetic_payload(family="molecular_vibronic_h2", strict_qpu_hh=False)
    payload["summary"].update(
        {
            "uses_future_exact_forecast_for_decision": False,
            "exact_forecast_guardrail_mode": "off",
            # Legacy name used by controller summaries. For strict observable
            # routes this may count local forecast vetoes; it is not an exact
            # decision input unless the future-exact flag or guardrail mode is
            # active.
            "exact_forecast_veto_count": 3,
            "exact_forecast_baseline_proposal_mode": "norm_locked_blend_v1",
        }
    )
    output_json = tmp_path / "strict_inactive_forecast_veto_count.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            study_profile=profile,
            t_final=2.0,
            num_times=3,
        ),
    )
    reasons = _invalid_reasons(metrics)

    assert metrics["strict_decision_contract_passed"] is True
    assert metrics["qpu_faithful_decisions_passed"] is True
    assert not any(reason.startswith("strict_purity_contract:summary.exact_forecast") for reason in reasons)


def test_strict_invalidation_uses_purity_and_measured_degradation_only() -> None:
    metrics = {
        "strict_qpu_faithful": True,
        "strict_qpu_hh": False,
        "strict_qpu_family": "hubbard",
        "qpu_faithful_decisions_expected": True,
        "qpu_faithful_decisions_passed": False,
        "strict_fail_closed": True,
        "strict_fail_closed_reason": "measured_baseline_error: ideal measurement unavailable",
        "exact_decision_checkpoints": 0,
        "reference_enabled": False,
        "reference_mode": "off",
        "decision_noise_mode": "ideal",
        "trajectory_reached_final_time": True,
        "trajectory_reached_expected_rows": True,
        "full_horizon_gate_passed": True,
        "mean_rho_miss": 0.02,
        "max_rho_miss": 0.04,
        "strict_degraded_fraction": 0.5,
        "strict_degraded_checkpoint_count": 1,
        "final_runtime_parameter_count": 3,
        # Exact-only fields are intentionally bad and must not drive strict invalidation.
        "pair_mae_over_exact_span": float("nan"),
        "epsilon_osc_pair": float("nan"),
        "dominant_peak_abs_omega_error": float("nan"),
        "mean_abs_energy_total_error": 999.0,
        "min_fidelity_exact": -999.0,
    }

    reasons = _invalid_reasons(metrics)

    assert "strict_purity:qpu_faithful_decisions_failed" in reasons
    assert "strict_fail_closed:measured_baseline_error: ideal measurement unavailable" in reasons
    assert "strict_degraded_checkpoints:1" in reasons
    assert not any("pair_mae" in reason for reason in reasons)
    assert not any("epsilon_osc" in reason for reason in reasons)
    assert not any("fidelity_exact" in reason for reason in reasons)
    assert not any("energy_total_error" in reason for reason in reasons)


def test_strict_evaluation_delegates_to_generic_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = realtime_optuna._STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE
    called: dict[str, object] = {}

    def _fake_generic_run(args: object) -> dict[str, object]:
        called["args"] = args
        payload = _strict_synthetic_payload(family="hubbard", strict_qpu_hh=False)
        Path(str(getattr(args, "output_json"))).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        return payload

    def _forbidden_bundle(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("strict Optuna trial used exact-v1 override seam")

    monkeypatch.setattr(realtime_optuna, "run_generic_realtime_from_args", _fake_generic_run)
    monkeypatch.setattr(
        realtime_optuna,
        "_build_controller_bundle_with_optuna_overrides",
        _forbidden_bundle,
    )

    obs = _evaluate_trial(
        trial_number=0,
        params=_baseline_trial_params(profile=profile),
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            study_profile=profile,
            t_final=2.0,
            num_times=3,
        ),
        objective_weights=ObjectiveWeights(),
        validity_gates=ValidityGates(),
        output_dir=tmp_path,
    )

    assert obs.status == "completed"
    assert called["args"] is not None
    assert getattr(called["args"], "checkpoint_controller_strict_qpu_faithful") is True
    assert obs.metrics["strict_qpu_family"] == "hubbard"


@pytest.mark.parametrize(
    "profile",
    [
        "append_live_guard_l2_v1",
        _FULL_SURFACE_PROFILE,
        "high_amp_guarded_d_shape_v1",
    ],
)
def test_actual_optuna_realtime_parser_accepts_advanced_exact_v1_profiles(
    profile: str,
) -> None:
    tokens = _build_realtime_tokens(
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json"), study_profile=profile),
        params=_baseline_trial_params(profile=profile),
        output_json=Path("out.json"),
        run_tag="advanced_exact_parser",
    )

    args = realtime_optuna.build_realtime_parser().parse_args(tokens)

    assert args.checkpoint_controller_mode == "exact_v1"
    assert args.checkpoint_controller_reference_mode == "benchmark_exact"


def test_exact_v1_profile_remains_diagnostic_fast_lane_for_generic_artifacts() -> None:
    tokens = _build_realtime_tokens(
        base_cfg=BaseRunConfig(
            artifact_json=Path("hubbard_artifact.json"),
            study_profile="generic_l2_exact_v1",
            append_pool_family="match_replay",
            exact_steps_multiplier=1,
        ),
        params=_baseline_trial_params(profile="generic_l2_exact_v1"),
        output_json=Path("out.json"),
        run_tag="generic_exact_diag_trial",
    )

    assert "--checkpoint-controller-strict-qpu-faithful" not in tokens
    assert tokens[tokens.index("--checkpoint-controller-mode") + 1] == "exact_v1"
    assert tokens[tokens.index("--checkpoint-controller-reference-mode") + 1] == "benchmark_exact"
    assert "--checkpoint-controller-exact-forecast-horizon-steps" in tokens
    assert "--checkpoint-controller-exact-forecast-horizon-weights" in tokens
    args = build_generic_realtime_parser().parse_args(tokens)
    assert args.checkpoint_controller_mode == "exact_v1"
    assert args.checkpoint_controller_reference_mode == "benchmark_exact"
    assert args.append_pool_family == "match_replay"


@pytest.mark.parametrize(
    ("fixture_name", "requested_mode", "expected_family"),
    [
        ("spin_boson_realtime_seed.json", "pair_difference", "spin_boson"),
        ("ttprime_hubbard_realtime_seed.json", "staggered", "ttprime_hubbard"),
    ],
)
def test_generic_exact_v1_route_context_locks_nonhh_primary_density_mode(
    tmp_path: Path,
    fixture_name: str,
    requested_mode: str,
    expected_family: str,
) -> None:
    base_cfg = BaseRunConfig(
        artifact_json=REPO_ROOT / "test_support" / "fixtures" / fixture_name,
        study_profile="generic_l2_exact_v1",
        enable_drive=True,
        drive_A=0.2,
        t_final=0.2,
        num_times=5,
    )
    route_ctx = realtime_optuna._resolve_optuna_route_context(base_cfg)

    assert route_ctx is not None
    assert route_ctx.problem_family == expected_family
    assert route_ctx.locked_primary_density_target_mode == "auto"
    locked_space = realtime_optuna._apply_route_context_to_search_space(
        _search_space("generic_l2_exact_v1"),
        route_ctx,
    )
    assert locked_space["primary_density_target_mode"] == ["auto"]
    enqueue = realtime_optuna._trial_to_enqueue_params(
        TrialParams(primary_density_target_mode=requested_mode),
        profile="generic_l2_exact_v1",
        route_ctx=route_ctx,
    )
    assert enqueue["primary_density_target_mode"] == "auto"

    tokens = _build_realtime_tokens(
        base_cfg=base_cfg,
        params=TrialParams(primary_density_target_mode=requested_mode),
        output_json=tmp_path / "out.json",
        run_tag=f"{expected_family}_route_lock",
        route_ctx=route_ctx,
    )

    flag = "--checkpoint-controller-exact-forecast-primary-density-target-mode"
    assert tokens[tokens.index(flag) + 1] == "auto"
    args = build_generic_realtime_parser().parse_args(tokens)
    assert args.checkpoint_controller_exact_forecast_primary_density_target_mode == "auto"


def test_trial_metrics_soft_fallback_is_physical_and_not_invalid_by_itself(tmp_path: Path) -> None:
    payload = _synthetic_payload()
    row = dict(payload["trajectory"][1])
    row.update(
        {
            "action_kind": "stay",
            "trajectory_sample_kind": "state_sample",
            "advances_time": True,
            "controller_lane": "append",
            "high_miss_no_admit_soft_fallback": True,
            "high_miss_no_admit_soft_fallback_policy": "bounded_stay_advance",
            "high_miss_no_admit_soft_fallback_reason": "bounded_high_miss_no_admit_stay_advance",
            "repair_no_admit_diagnostics": {
                "controller_lane": "append",
                "strict_no_admit_reason": "no_confirmed_candidates",
                "no_admit_resolution": "bounded_stay_advance",
                "no_admit_resolution_advances_time": True,
            },
        }
    )
    payload["trajectory"][1] = row
    payload["summary"] = {
        **dict(payload["summary"]),
        "status": "completed",
        "stay_count": 1,
        "high_miss_no_admit_soft_fallback_count": 1,
        "high_miss_no_admit_soft_fallback_warning_count": 1,
        "ordinary_stay_count": 0,
        "high_miss_no_admit_soft_fallback_reason_counts": {
            "bounded_high_miss_no_admit_stay_advance": 1,
        },
    }
    output_json = tmp_path / "soft_fallback_result.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    analysis, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json"), pair="auto", t_final=2.0, num_times=3),
    )
    reasons = _invalid_reasons(metrics)

    assert analysis["metadata"]["high_miss_no_admit_soft_fallback_count"] == 1
    assert analysis["metadata"]["high_miss_count"] == 1
    assert analysis["metadata"]["high_miss_no_admit_count"] == 1
    assert metrics["high_miss_no_admit_soft_fallback_count"] == 1
    assert metrics["high_miss_no_admit_soft_fallback_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["ordinary_stay_count"] == 0
    assert metrics["soft_fallback_reason_counts"] == {"bounded_high_miss_no_admit_stay_advance": 1}
    assert metrics["high_miss_count"] == 1
    assert metrics["high_miss_no_admit_count"] == 1
    assert metrics["high_miss_no_admit_reason_counts"] == {"no_confirmed_candidates": 1}
    assert metrics["high_miss_no_admit_resolution_counts"] == {"bounded_stay_advance": 1}
    assert metrics["first_bad_high_miss_no_admit_checkpoint_diagnostic"]["checkpoint_index"] is None
    assert metrics["repair_event_row_count"] == 0
    assert metrics["trajectory_state_sample_count"] == 3
    assert not any("soft_fallback" in reason for reason in reasons)


def test_trial_metrics_terminal_repair_retry_exhaustion_is_invalid(tmp_path: Path) -> None:
    payload = _short_early_stop_payload()
    payload["summary"] = {
        **dict(payload["summary"]),
        "status": "stopped_early",
        "early_stop_reason": "repair_retry_exhausted_high_miss_no_admit",
    }
    row = dict(payload["trajectory"][0])
    row.update(
        {
            "action_kind": "repair_miss",
            "trajectory_sample_kind": "repair_event",
            "advances_time": False,
            "repair_terminal": True,
            "repair_failure_reason": "repair_retry_exhausted_high_miss_no_admit",
        }
    )
    payload["trajectory"] = [row]
    output_json = tmp_path / "retry_exhausted_result.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json"), pair="auto"),
    )
    reasons = _invalid_reasons(metrics)

    assert metrics["repair_event_row_count"] == 1
    assert metrics["trajectory_state_sample_count"] == 0
    assert metrics["early_stop_reason"] == "repair_retry_exhausted_high_miss_no_admit"
    assert "trajectory_early_stop:repair_retry_exhausted_high_miss_no_admit" in reasons


def test_render_spectrum_pdf_from_synthetic_analysis(tmp_path: Path) -> None:
    payload = _synthetic_payload()
    output_json = tmp_path / "result_for_pdf.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    analysis, _ = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json"), pair="auto"),
    )

    output_pdf = tmp_path / "spectra.pdf"
    render_spectrum_pdf(analysis, output_pdf=output_pdf, max_harmonic=3)

    assert output_pdf.exists()
    assert output_pdf.read_bytes().startswith(b"%PDF")


def test_trial_metrics_from_payload_supports_objective_window_overlay(tmp_path: Path) -> None:
    payload = _synthetic_payload()
    output_json = tmp_path / "result_window.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            pair="auto",
            objective_window_start=1.0,
            objective_window_end=2.0,
            spectra_detrend="linear",
            spectra_window="hann",
        ),
    )
    assert metrics["objective_window_num_samples"] == 2
    assert metrics["pair_mae_over_exact_span"] == pytest.approx(0.5)
    assert metrics["objective_pair_mae_over_exact_span"] == pytest.approx(0.75)
    assert metrics["objective_mean_abs_site_occupations_error"] == pytest.approx((0.1 + 0.2) / 2.0)


def test_objective_value_prefers_objective_window_metrics() -> None:
    weights = ObjectiveWeights(
        pair_mae_over_exact_span=1.0,
        epsilon_osc_pair=0.0,
        dominant_peak_abs_omega_error=0.0,
        pair_corr_defect=0.0,
        mean_abs_site_occupations_error=0.0,
        mean_abs_energy_total_error=0.0,
        fidelity_defect=0.0,
        final_runtime_parameter_count=0.0,
        append_count=0.0,
        prune_count=0.0,
    )
    metrics = {
        "pair_mae_over_exact_span": 0.5,
        "objective_pair_mae_over_exact_span": 0.75,
        "epsilon_osc_pair": 0.0,
        "dominant_peak_abs_omega_error": 0.0,
        "pair_corr": 1.0,
        "mean_abs_site_occupations_error": 0.0,
        "mean_abs_energy_total_error": 0.0,
        "min_fidelity_exact": 1.0,
        "final_runtime_parameter_count": 0.0,
        "append_count": 0.0,
        "prune_count": 0.0,
    }
    assert _objective_value(metrics, weights) == pytest.approx(0.75)


def test_invalid_reasons_require_finite_quality_metrics() -> None:
    reasons = _invalid_reasons(
        {
            "pair_mae_over_exact_span": float("nan"),
            "epsilon_osc_pair": 0.2,
            "dominant_peak_abs_omega_error": 0.1,
            "mean_abs_site_occupations_error": 0.02,
            "mean_abs_energy_total_error": 0.01,
            "min_fidelity_exact": 0.95,
            "final_runtime_parameter_count": 4,
        }
    )
    assert "missing_or_nonfinite:pair_mae_over_exact_span" in reasons


def test_invalid_reasons_respect_energy_gate() -> None:
    reasons = _invalid_reasons(
        {
            "pair_mae_over_exact_span": 0.1,
            "epsilon_osc_pair": 0.2,
            "dominant_peak_abs_omega_error": 0.0,
            "mean_abs_site_occupations_error": 0.05,
            "mean_abs_energy_total_error": 0.3,
            "min_fidelity_exact": 0.95,
            "final_runtime_parameter_count": 4,
        },
        gates=ValidityGates(max_mean_abs_energy_total_error=0.2),
    )
    assert "above_gate:mean_abs_energy_total_error>0.2" in reasons


def test_invalid_reasons_respect_final_energy_fidelity_and_total_occupation_gates() -> None:
    reasons = _invalid_reasons(
        {
            "pair_mae_over_exact_span": 0.1,
            "epsilon_osc_pair": 0.2,
            "dominant_peak_abs_omega_error": 0.0,
            "mean_abs_site_occupations_error": 0.05,
            "mean_abs_energy_total_error": 0.2,
            "final_abs_energy_total_error": 0.3,
            "mean_total_occupation_abs_error": 0.06,
            "min_fidelity_exact": 0.4,
            "final_runtime_parameter_count": 4,
        },
        gates=ValidityGates(
            max_mean_abs_energy_total_error=0.25,
            max_final_abs_energy_total_error=0.25,
            min_fidelity_exact=0.5,
            max_mean_total_occupation_abs_error=0.05,
        ),
    )
    assert "above_gate:final_abs_energy_total_error>0.25" in reasons
    assert "below_gate:min_fidelity_exact<0.5" in reasons
    assert "above_gate:mean_total_occupation_abs_error>0.05" in reasons


def test_invalid_reasons_respect_generic_exact_primary_observable_relative_gate() -> None:
    metrics = {
        "generic_exact_v1_family_objective": True,
        "primary_observable_mae_over_exact_span": 0.7,
        "mean_abs_site_occupations_error": 0.01,
        "mean_abs_energy_total_error": 0.01,
        "final_abs_energy_total_error": 0.01,
        "mean_total_occupation_abs_error": 0.01,
        "min_fidelity_exact": 0.99,
        "final_runtime_parameter_count": 4,
        "trajectory_reached_final_time": True,
        "trajectory_reached_expected_rows": True,
        "full_horizon_gate_passed": True,
    }
    reasons = _invalid_reasons(
        metrics,
        gates=ValidityGates(max_primary_observable_mae_over_exact_span=0.5),
    )
    assert "above_gate:primary_observable_mae_over_exact_span>0.5" in reasons


def test_append_prune_profile_uses_generic_exact_v1_family_objective(tmp_path: Path) -> None:
    payload = _synthetic_payload()
    output_json = tmp_path / "append_prune_result.json"
    output_json.write_text(json.dumps(payload), encoding="utf-8")

    _analysis, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            study_profile="append_prune_noharm_l2_v1",
            pair="auto",
        ),
    )

    assert metrics["generic_exact_v1_family_objective"] is True
    assert "missing_or_nonfinite:primary_observable_mae_over_exact_span" not in _invalid_reasons(metrics)


def test_invalid_reasons_reject_exact_span_gate_for_strict_qpu_faithful_metrics() -> None:
    metrics = {
        "strict_qpu_faithful": True,
        "qpu_faithful_decisions_expected": True,
        "qpu_faithful_decisions_passed": True,
        "strict_decision_contract_passed": True,
        "strict_fail_closed": False,
        "exact_decision_checkpoints": 0,
        "reference_enabled": False,
        "reference_mode": "off",
        "decision_noise_mode": "ideal",
        "non_ideal_decision_noise_count": 0,
        "trajectory_reached_final_time": True,
        "trajectory_reached_expected_rows": True,
        "full_horizon_gate_passed": True,
        "mean_rho_miss": 0.0,
        "max_rho_miss": 0.0,
        "strict_degraded_fraction": 0.0,
        "final_runtime_parameter_count": 4,
    }
    reasons = _invalid_reasons(
        metrics,
        gates=ValidityGates(max_primary_observable_mae_over_exact_span=0.5),
    )
    assert "strict_purity:exact_span_gate_configured" in reasons


def test_invalid_reasons_accept_controller_declared_stable_early_stop() -> None:
    metrics = {
        "generic_exact_v1_family_objective": True,
        "primary_observable_mae_over_exact_span": 0.2,
        "mean_abs_site_occupations_error": 0.01,
        "mean_abs_energy_total_error": 0.01,
        "min_fidelity_exact": 0.99,
        "final_runtime_parameter_count": 4,
        "early_stop_reason": "progress_observables_stable:site_span=0.0001",
        "full_horizon_early_stop_reason": "progress_observables_stable:site_span=0.0001",
        "trajectory_reached_final_time": False,
        "trajectory_reached_expected_rows": False,
        "full_horizon_gate_passed": True,
        "full_horizon_successful_early_stop": True,
        "full_horizon_completion_kind": "stable_early_stop",
        "full_horizon_gate_reason": "stable_early_stop:progress_observables_stable:site_span=0.0001",
    }
    reasons = _invalid_reasons(metrics)
    assert not any(reason.startswith("trajectory_early_stop:") for reason in reasons)
    assert "trajectory_incomplete:final_time" not in reasons
    assert "trajectory_incomplete:row_count" not in reasons
    assert not any(reason.startswith("full_horizon_gate:") for reason in reasons)


def test_objective_value_penalizes_quality_and_cost() -> None:
    weights = ObjectiveWeights(
        pair_mae_over_exact_span=1.0,
        epsilon_osc_pair=0.5,
        dominant_peak_abs_omega_error=0.25,
        pair_corr_defect=0.25,
        mean_abs_site_occupations_error=0.25,
        mean_abs_energy_total_error=0.1,
        fidelity_defect=0.1,
        final_runtime_parameter_count=0.01,
        append_count=0.05,
        prune_count=0.05,
    )
    baseline = {
        "pair_mae_over_exact_span": 0.2,
        "epsilon_osc_pair": 0.3,
        "dominant_peak_abs_omega_error": 0.1,
        "pair_corr": 0.9,
        "mean_abs_site_occupations_error": 0.04,
        "mean_abs_energy_total_error": 0.02,
        "min_fidelity_exact": 0.97,
        "final_runtime_parameter_count": 4,
        "append_count": 1,
        "prune_count": 0,
    }
    worse = dict(baseline)
    worse["pair_mae_over_exact_span"] = 0.4
    worse["append_count"] = 3
    worse["prune_count"] = 1
    assert _objective_value(worse, weights) > _objective_value(baseline, weights)


def test_baseline_trial_params_match_current_exact_v1_anchor() -> None:
    params = _baseline_trial_params()
    assert params.horizon_mode == "lead2"
    assert params.include_tangent_secant_proposal is False
    assert params.trust_radius == pytest.approx(0.75)
    assert params.signed_energy_lead_limit == pytest.approx(1.0)
    assert params.primary_density_target_mode == "auto"


def test_high_amp_profile_exposes_d_shape_axes() -> None:
    space = _search_space(profile="high_amp_d_shape_v1")
    assert space["horizon_mode"] == ["high_amp_lead4"]
    assert "d_shape_barrier_v1" in space["guardrail_mode"]
    assert True in space["below_floor_energy_safe_d_shape_escape"]


def test_expanded_suggest_trial_params_skips_dead_knobs() -> None:
    class _FakeTrial:
        def __init__(self, picks: dict[str, object]):
            self.picks = picks
            self.calls: list[str] = []

        def suggest_categorical(self, name: str, choices: list[object]) -> object:
            self.calls.append(name)
            value = self.picks[name]
            assert value in choices
            return value

    trial = _FakeTrial(
        {
            "horizon_mode": "high_amp_lead4",
            "step_scale_mode": "high_amp_dense",
            "blend_weight_mode": "high_amp",
            "gain_scale_mode": "high_amp",
            "baseline_step_refine_rounds": 2,
            "include_tangent_secant_proposal": False,
            "baseline_proposal_mode": "norm_locked_blend_v1",
            "primary_density_target_mode": "auto",
            "miss_threshold": 0.05,
            "gain_ratio_threshold": 0.02,
            "append_margin_abs": 1.0e-6,
            "shortlist_size": 4,
            "shortlist_fraction": 0.15,
            "active_window_size": 3,
            "max_probe_positions": 4,
            "regularization_lambda": 1.0e-8,
            "candidate_regularization_lambda": 1.0e-8,
            "pinv_rcond": 1.0e-10,
            "compile_penalty_weight": 0.05,
            "measurement_penalty_weight": 0.02,
            "directional_penalty_weight": 0.01,
            "confirm_score_mode": "exact_gain_ratio",
            "primary_density_weight": 2.0,
            "site_weight": 0.0,
            "energy_weight": 0.0,
            "density_slope_weight": 2.0,
            "density_sign_lag_weight": 0.25,
            "drive_harmonic_weight": 0.0,
            "density_first_target_gain_floor": 0.02,
            "below_floor_probe_target_gain_floor": 0.03,
            "fidelity_loss_tol": 0.01,
            "abs_energy_error_increase_tol": 0.02,
            "total_occupation_error_increase_tol": 0.01,
            "guardrail_mode": "d_shape_barrier_v1",
            "below_floor_energy_safe_d_shape_escape": False,
            "below_floor_energy_safe_turn_escape": False,
            "d_shape_turn_window_abs_activation": 0.04,
            "d_shape_outside_turn_below_floor_probe_stall_threshold": 0,
            "d_shape_pre_turn_shadow_bridge": False,
        }
    )
    params = realtime_optuna._suggest_trial_params(
        trial,
        profile="high_amp_guarded_d_shape_expanded_v1",
    )
    assert params.include_tangent_secant_proposal is False
    assert params.trust_radius == pytest.approx(1.0)
    assert params.signed_energy_lead_limit == pytest.approx(3.5)
    assert params.confirm_score_mode == "exact_gain_ratio"
    assert params.confirm_compress_fraction == pytest.approx(0.5)
    assert params.confirm_compress_min_modes == 1
    assert params.confirm_compress_max_modes == 8
    assert "trust_radius" not in trial.calls
    assert "signed_energy_lead_limit" not in trial.calls
    assert "confirm_compress_fraction" not in trial.calls
    assert "confirm_compress_min_modes" not in trial.calls
    assert "confirm_compress_max_modes" not in trial.calls


def test_high_amp_baseline_round_trip_through_current_parser() -> None:
    base = BaseRunConfig(
        artifact_json=Path("artifact.json"),
        study_profile="high_amp_d_shape_v1",
        enable_drive=True,
        drive_A=1.5,
        drive_omega=1.2,
        drive_tbar=4.0,
        exact_steps_multiplier=10,
    )
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=_baseline_trial_params(profile="high_amp_d_shape_v1"),
        output_json=Path("out.json"),
        run_tag="trial_high_amp",
    )
    args = build_realtime_parser().parse_args(tokens)
    assert args.checkpoint_controller_prune_mode == "exact_local_v1"
    assert args.checkpoint_controller_exact_forecast_guardrail_mode == "off"
    assert args.checkpoint_controller_exact_v1_sign_lag_window_target_gain_floor == pytest.approx(0.005)
    assert args.checkpoint_controller_exact_forecast_horizon_weights == "8.0,4.0,2.0,1.0"
    assert args.checkpoint_controller_exact_forecast_baseline_gain_scales == "1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4"


def test_guarded_profile_locks_guardrail_axes() -> None:
    space = _search_space(profile="high_amp_guarded_d_shape_v1")
    assert space["guardrail_mode"] == ["d_shape_barrier_v1"]
    assert space["below_floor_energy_safe_d_shape_escape"] == [False]
    assert space["density_first_target_gain_floor"] == [0.015, 0.02, 0.025]
    assert space["postcross_wrong_sign_activation"] == [0.0, 0.05, 0.1]


def test_live_guarded_profile_locks_current_lane_knobs() -> None:
    space = _search_space(profile="high_amp_guarded_d_shape_live_v1")
    assert space["guardrail_mode"] == ["d_shape_barrier_v1"]
    assert space["below_floor_energy_safe_d_shape_escape"] == [False]
    assert space["below_floor_energy_safe_turn_escape"] == [False]
    assert space["d_shape_turn_window_abs_activation"] == [0.04]
    assert space["d_shape_outside_turn_below_floor_probe_stall_threshold"] == [0]
    assert space["d_shape_pre_turn_shadow_bridge"] == [False]
    assert space["drive_harmonic_weight"] == [0.0, 0.1, 0.25]


def test_turnwindow_profile_unlocks_window_geometry_knobs() -> None:
    space = _search_space(profile="high_amp_guarded_d_shape_turnwindow_v1")
    assert space["guardrail_mode"] == ["d_shape_barrier_v1"]
    assert space["below_floor_energy_safe_d_shape_escape"] == [False]
    assert space["below_floor_energy_safe_turn_escape"] == [False]
    assert space["d_shape_turn_window_abs_activation"] == [0.02, 0.03, 0.04, 0.05, 0.06]
    assert space["d_shape_outside_turn_below_floor_probe_stall_threshold"] == [0, 1, 2, 3]
    assert space["d_shape_pre_turn_shadow_bridge"] == [False, True]
    assert space["drive_harmonic_weight"] == [0.0, 0.1, 0.25]


def test_expanded_profile_unlocks_structural_knobs() -> None:
    space = _search_space(profile="high_amp_guarded_d_shape_expanded_v1")
    assert space["baseline_proposal_mode"] == [
        "norm_locked_blend_v1",
        "anticipatory_drive_basis_v1",
    ]
    assert space["include_tangent_secant_proposal"] == [False, True]
    assert space["primary_density_target_mode"] == ["auto", "pair_difference", "staggered"]
    assert space["append_margin_abs"] == [1.0e-6, 1.0e-5, 5.0e-5]
    assert space["shortlist_size"] == [4, 6, 8]
    assert space["confirm_score_mode"] == ["compressed_whitened_v1", "exact_gain_ratio"]
    assert space["confirm_compress_fraction"] == [0.25, 0.5, 0.75]
    assert space["regularization_lambda"] == [1.0e-10, 1.0e-8, 1.0e-6]
    assert space["compile_penalty_weight"] == [0.0, 0.05, 0.1]


def test_full_surface_profile_exposes_explicit_controller_lists_and_new_algorithm_seams() -> None:
    space = _search_space(profile=_FULL_SURFACE_PROFILE)
    assert "oracle_selection_policy" in space
    assert "candidate_step_scales" in space
    assert "baseline_blend_weights" in space
    assert "baseline_gain_scales" in space
    assert "horizon_spec" in space
    assert "tracking_staggered_weight" in space
    assert "single_surface_commit_law" in space
    assert "prune_miss_threshold" in space
    assert "prune_loss_threshold" in space
    assert "primary_density_scale_floor" in space
    assert "prune_stagnation_alpha" in space
    assert "motion_calm_direction_cosine_threshold" in space
    assert "motion_kink_oracle_budget_scale" in space
    assert "reconstruction_tol" in space
    assert "grouping_mode" in space
    assert space["guardrail_mode"] == ["d_shape_barrier_v1", "fidelity_first_barrier_v1"]


def test_full_surface_baseline_exposes_hidden_exact_controller_defaults() -> None:
    params = _baseline_trial_params(profile=_FULL_SURFACE_PROFILE)
    assert params.primary_density_scale_floor == pytest.approx(1.0e-6)
    assert params.energy_total_scale_floor == pytest.approx(1.0e-6)
    assert params.prune_protection_steps == 2
    assert params.prune_stagnation_alpha == pytest.approx(0.5)
    assert params.motion_calm_direction_cosine_threshold == pytest.approx(0.98)
    assert params.motion_kink_rate_change_ratio_threshold == pytest.approx(0.5)
    assert params.motion_kink_oracle_budget_scale == pytest.approx(2.0)
    assert params.reconstruction_tol == pytest.approx(1.0e-10)
    assert params.grouping_mode == "qwc_basis_cover_reuse"


def test_append_prune_noharm_profile_exposes_append_cleanup_knobs() -> None:
    space = _search_space(profile="append_prune_noharm_l2_v1")
    assert space["append_no_harm_guard_enabled"] == [True, False]
    assert space["prune_mode"] == ["off", "exact_local_v1"]
    assert space["prune_appended_origin_target_policy"] == ["append_only"]
    assert space["prune_appended_origin_bias_enabled"] == [True]
    assert space["integrator_condition_max"] == [1.0e10, 1.0e12, 1.0e14]
    assert False in space["include_tangent_secant_proposal"]
    params = _baseline_trial_params(profile="append_prune_noharm_l2_v1")
    assert params.high_miss_no_admit_policy == "bounded_stay_advance"
    assert params.miss_persistence_spec == "3:3"
    assert params.append_no_harm_guard_enabled is True
    assert params.prune_appended_origin_target_policy == "append_only"


def test_append_prune_recoverability_profile_uses_schur_projection_without_prune_reward() -> None:
    profile = "append_prune_recoverability_l2_v1"
    space = _search_space(profile=profile)
    assert space["prune_mode"] == ["off", "schur_projected_shadow_v1"]
    assert space["prune_appended_origin_target_policy"] == ["prefer_append", "bias_only"]
    assert space["integrator_condition_max"] == [1.0e10, 1.0e12, 1.0e14]
    assert space["prune_persistence_required"] == [2]
    params = _baseline_trial_params(profile=profile)
    assert params.prune_mode == "schur_projected_shadow_v1"
    assert params.prune_appended_origin_target_policy == "prefer_append"
    assert params.prune_persistence_window == 2
    assert params.prune_persistence_required == 2
    assert realtime_optuna._default_objective_weights(profile).prune_count == pytest.approx(0.0)

    tokens = _build_realtime_tokens(
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json"), study_profile=profile),
        params=params,
        output_json=Path("out.json"),
        run_tag="recoverability_roundtrip",
    )
    args = build_realtime_parser().parse_args(tokens)
    cfg = build_controller_config(args)
    assert cfg.prune_mode == "schur_projected_shadow_v1"
    assert cfg.prune_differential_miss_tol == pytest.approx(0.01)
    assert cfg.prune_projection_mode == "state_tangent_ls_v1"
    assert cfg.prune_persistence_window == 2
    assert cfg.prune_persistence_required == 2


def test_strict_append_prune_aggressive_profile_forces_prune_and_rewards_pruning() -> None:
    profile = "strict_qpu_faithful_append_prune_aggressive_v1"
    space = _search_space(profile=profile)
    assert space["prune_mode"] == ["schur_projected_shadow_v1"]
    assert space["prune_persistence_required"] == [1]
    assert space["prune_initial_scaffold_grace_steps"] == [0]
    assert space["prune_appended_origin_target_policy"] == ["bias_only", "prefer_append"]
    params = _baseline_trial_params(profile=profile)
    assert params.prune_mode == "schur_projected_shadow_v1"
    assert params.prune_miss_threshold == pytest.approx(0.05)
    assert params.prune_loss_threshold == pytest.approx(0.08)
    assert params.prune_persistence_window == 1
    assert params.prune_persistence_required == 1
    weights = realtime_optuna._default_objective_weights(profile)
    assert weights.prune_count == pytest.approx(1.0)
    assert weights.append_count == pytest.approx(0.03)
    assert weights.mean_abs_energy_total_error == pytest.approx(0.0)
    seeds = _profile_seed_trials(profile=profile)
    assert len(seeds) >= 4
    assert all(seed.prune_mode == "schur_projected_shadow_v1" for seed in seeds)

    tokens = _build_realtime_tokens(
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json"), study_profile=profile),
        params=params,
        output_json=Path("out.json"),
        run_tag="aggressive_roundtrip",
    )
    args = build_realtime_parser().parse_args(tokens)
    cfg = build_controller_config(args)
    assert cfg.prune_mode == "schur_projected_shadow_v1"
    assert cfg.prune_appended_origin_target_policy == "bias_only"
    assert cfg.prune_initial_scaffold_grace_steps == 0


def test_strict_objective_ignores_diagnostic_exact_physical_errors() -> None:
    weights = realtime_optuna._default_objective_weights(
        "strict_qpu_faithful_append_prune_recoverability_v1"
    )
    base_metrics = {
        "strict_qpu_faithful": True,
        "strict_fail_closed": False,
        "qpu_faithful_decisions_passed": True,
        "strict_measured_degradation_score": 0.2,
        "max_rho_miss": 0.4,
        "mean_predicted_displacement": 0.01,
        "strict_degraded_fraction": 0.0,
        "high_miss_no_admit_fraction": 0.0,
        "high_miss_no_admit_soft_fallback_fraction": 0.0,
        "final_runtime_parameter_count": 50,
        "append_count": 3,
        "prune_count": 0,
        "append_opportunity_count": 3,
        "proposed_append_count": 3,
        "prune_opportunity_count": 0,
        "prune_candidate_checkpoint_count": 0,
        "integrator_rk4_count": 320,
    }
    physically_good = {
        **base_metrics,
        "diagnostic_mean_abs_energy_total_error": 1.0e-3,
        "diagnostic_mean_abs_site_occupations_error": 2.0e-3,
    }
    physically_bad = {
        **base_metrics,
        "diagnostic_mean_abs_energy_total_error": 1.0,
        "diagnostic_mean_abs_site_occupations_error": 1.0,
    }
    assert _objective_value(physically_good, weights) == pytest.approx(
        _objective_value(physically_bad, weights)
    )


def test_best_diagnostic_physical_trial_is_report_only_and_separate_from_objective() -> None:
    poor_objective_good_physics = realtime_optuna.TrialObservation(
        trial_number=2,
        params={"gain_ratio_threshold": 0.001},
        objective=10.0,
        status="completed",
        metrics={
            "full_horizon_gate_passed": True,
            "diagnostic_mean_abs_energy_total_error": 1.0e-3,
            "diagnostic_max_abs_energy_total_error": 2.0e-3,
            "diagnostic_mean_abs_site_occupations_error": 3.0e-3,
            "diagnostic_max_abs_site_occupations_error": 4.0e-3,
        },
        invalid_reasons=[],
    )
    good_objective_bad_physics = realtime_optuna.TrialObservation(
        trial_number=1,
        params={"gain_ratio_threshold": 0.01},
        objective=1.0,
        status="completed",
        metrics={
            "full_horizon_gate_passed": True,
            "diagnostic_mean_abs_energy_total_error": 0.1,
            "diagnostic_max_abs_energy_total_error": 0.2,
            "diagnostic_mean_abs_site_occupations_error": 0.3,
            "diagnostic_max_abs_site_occupations_error": 0.4,
        },
        invalid_reasons=[],
    )
    best = realtime_optuna._best_by_diagnostic_physical_score(
        [good_objective_bad_physics, poor_objective_good_physics]
    )
    assert best is not None
    assert best["trial_number"] == 2
    assert best["objective"] == pytest.approx(10.0)
    assert "not strict Optuna online feedback" in best["selection_note"]


def test_append_prune_noharm_profile_seed_trials_cover_useful_append_diagnostic() -> None:
    seeds = _profile_seed_trials(profile="append_prune_noharm_l2_v1")
    assert any(seed.append_no_harm_guard_enabled is False for seed in seeds)
    assert any(seed.prune_mode == "off" for seed in seeds)
    assert all(
        seed.prune_appended_origin_target_policy in {"append_only", None}
        for seed in seeds
    )


def test_append_live_guard_profile_is_narrowed_around_valid_append_regime() -> None:
    space = _search_space(profile="append_live_guard_l2_v1")

    assert space["horizon_mode"] == ["lead2"]
    assert space["step_scale_mode"] == ["default", "wide"]
    assert "high_amp_dense" not in space["step_scale_mode"]
    assert space["append_no_harm_guard_enabled"] == [False, True]
    assert 1.0 not in space["append_no_harm_condition_ratio_cap"]
    assert space["prune_mode"] == ["off", "exact_local_v1"]
    assert space["prune_appended_origin_target_policy"] == ["append_only"]

    params = _baseline_trial_params(profile="append_live_guard_l2_v1")
    assert params.append_no_harm_guard_enabled is False
    assert params.prune_mode == "off"
    assert params.high_miss_no_admit_policy == "bounded_stay_advance"


def test_append_live_guard_profile_seed_trials_compare_guard_and_prune_modes() -> None:
    seeds = _profile_seed_trials(profile="append_live_guard_l2_v1")

    assert any(seed.append_no_harm_guard_enabled is False for seed in seeds)
    assert any(seed.append_no_harm_guard_enabled is True for seed in seeds)
    assert any(seed.prune_mode == "off" for seed in seeds)
    assert any(seed.prune_mode == "exact_local_v1" for seed in seeds)
    assert all(
        seed.prune_appended_origin_target_policy in {"append_only", None}
        for seed in seeds
    )


def test_hidden_exact_v1_cfg_overrides_collect_nonparser_fields() -> None:
    params = TrialParams(
        primary_density_scale_floor=1.0e-8,
        density_slope_scale_floor=1.0e-4,
        prune_protection_steps=4,
        prune_appended_origin_target_policy="append_only",
        prune_appended_origin_grace_steps=3,
        prune_appended_origin_bias_scale=0.5,
        prune_appended_origin_bias_max_factor=1.0,
        prune_stagnation_alpha=0.75,
        motion_calm_direction_cosine_threshold=0.995,
        motion_kink_shortlist_bonus=4,
        reconstruction_tol=1.0e-12,
        grouping_mode="qwc_basis_cover_reuse",
    )
    overrides = realtime_optuna._hidden_exact_v1_cfg_overrides(params)
    assert overrides == {
        "exact_forecast_primary_density_scale_floor": pytest.approx(1.0e-8),
        "exact_forecast_density_slope_scale_floor": pytest.approx(1.0e-4),
        "prune_protection_steps": 4,
        "prune_appended_origin_target_policy": "append_only",
        "prune_appended_origin_grace_steps": 3,
        "prune_appended_origin_bias_scale": pytest.approx(0.5),
        "prune_appended_origin_bias_max_factor": pytest.approx(1.0),
        "prune_stagnation_alpha": pytest.approx(0.75),
        "motion_calm_direction_cosine_threshold": pytest.approx(0.995),
        "motion_kink_shortlist_bonus": 4,
        "reconstruction_tol": pytest.approx(1.0e-12),
        "grouping_mode": "qwc_basis_cover_reuse",
    }


def test_build_controller_bundle_with_optuna_overrides_uses_neutral_seed_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    args = build_realtime_parser().parse_args(
        [
            "--artifact-json",
            str(tmp_path / "missing_artifact.json"),
            "--output-json",
            str(tmp_path / "out.json"),
        ]
    )

    def _fake_build_controller_seed_from_args(
        seed_args: object,
        *,
        cfg: object | None = None,
    ) -> object:
        assert seed_args is args
        assert cfg is not None
        captured["cfg"] = cfg
        return SimpleNamespace(cfg=cfg, marker="neutral_seed")

    def _fake_finalize_controller_bundle_from_seed(
        finalize_args: object,
        *,
        seed: object,
        exact_reference_cache: dict[str, object] | None = None,
    ) -> dict[str, object]:
        assert finalize_args is args
        assert exact_reference_cache is None
        captured["seed"] = seed
        return {
            "loaded": "loaded",
            "cfg": getattr(seed, "cfg"),
            "oracle_config": None,
            "drive_config": None,
            "controller": "controller",
        }

    monkeypatch.setattr(
        realtime_optuna,
        "build_controller_seed_from_args",
        _fake_build_controller_seed_from_args,
    )
    monkeypatch.setattr(
        realtime_optuna,
        "finalize_controller_bundle_from_seed",
        _fake_finalize_controller_bundle_from_seed,
    )

    bundle = realtime_optuna._build_controller_bundle_with_optuna_overrides(
        args=args,
        params=TrialParams(
            primary_density_scale_floor=1.0e-8,
            prune_protection_steps=4,
        ),
    )

    cfg = captured["cfg"]
    assert getattr(cfg, "exact_forecast_primary_density_scale_floor") == pytest.approx(1.0e-8)
    assert getattr(cfg, "prune_protection_steps") == 4
    assert getattr(captured["seed"], "marker") == "neutral_seed"
    assert bundle["cfg"] is cfg
    assert bundle["controller"] == "controller"


def test_full_surface_exact_controller_coverage_excludes_only_mode_noise_and_tiers() -> None:
    source = textwrap.dedent(inspect.getsource(build_controller_config))
    func = ast.parse(source).body[0]
    calls = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RealtimeCheckpointConfig"
    ]
    assert len(calls) == 1
    parser_cfg_fields = {str(kw.arg) for kw in calls[0].keywords if kw.arg is not None}
    hidden_cfg_fields = set(realtime_optuna._HIDDEN_EXACT_V1_CFG_FIELD_MAP.values())
    covered_cfg_fields = parser_cfg_fields | hidden_cfg_fields
    config_fields = set(RealtimeCheckpointConfig.__dataclass_fields__.keys())
    excluded = {
        "mode",
        "tiers",
        *{name for name in config_fields if str(name).startswith("analytic_noise_")},
    }
    missing = sorted(config_fields - covered_cfg_fields - excluded)
    assert missing == []


@pytest.mark.parametrize(("profile", "guardrail_mode"), _PREFSITE_AUTOBASELINE_PROFILES)
def test_prefsite_autobaseline_profile_locks_repaired_surface(
    profile: str,
    guardrail_mode: str,
) -> None:
    space = _search_space(profile=profile)
    assert space["include_tangent_secant_proposal"] == [True]
    assert space["baseline_proposal_mode"] == ["norm_locked_blend_v1"]
    assert space["primary_density_target_mode"] == ["pair_difference"]
    assert space["guardrail_mode"] == [guardrail_mode]
    assert space["d_shape_turn_window_abs_activation"] == [0.04]
    assert space["d_shape_outside_turn_below_floor_probe_stall_threshold"] == [7]
    assert space["d_shape_pre_turn_shadow_bridge"] == [True]
    assert space["append_margin_abs"] == [1.0e-6, 1.0e-5, 5.0e-5]
    assert space["shortlist_size"] == [4, 6, 8]


def test_guarded_high_amp_baseline_round_trip_through_current_parser() -> None:
    base = BaseRunConfig(
        artifact_json=Path("artifact.json"),
        study_profile="high_amp_guarded_d_shape_v1",
        enable_drive=True,
        drive_A=1.5,
        drive_omega=1.2,
        drive_tbar=4.0,
        exact_steps_multiplier=10,
    )
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=_baseline_trial_params(profile="high_amp_guarded_d_shape_v1"),
        output_json=Path("out.json"),
        run_tag="trial_guarded",
    )
    assert "--checkpoint-controller-progress-observable-window" not in tokens
    assert "--checkpoint-controller-progress-early-stop-min-checkpoint" not in tokens
    assert "--checkpoint-controller-progress-early-stop-site-error-mean-max" not in tokens
    assert "--checkpoint-controller-exact-v1-postcross-compare-diag" not in tokens
    args = build_realtime_parser().parse_args(tokens)
    assert args.checkpoint_controller_prune_mode == "exact_local_v1"
    assert args.checkpoint_controller_exact_forecast_guardrail_mode == "d_shape_barrier_v1"
    assert args.checkpoint_controller_exact_v1_density_first_target_gain_floor == pytest.approx(0.02)
    assert args.checkpoint_controller_exact_v1_below_floor_probe_target_gain_floor == pytest.approx(0.03)
    assert args.checkpoint_controller_exact_v1_sign_lag_window_target_gain_floor == pytest.approx(0.005)
    assert args.checkpoint_controller_exact_v1_postcross_wrong_sign_activation == pytest.approx(0.0)


def test_guarded_profile_seed_trials_round_trip_through_current_parser() -> None:
    base = BaseRunConfig(
        artifact_json=Path("artifact.json"),
        study_profile="high_amp_guarded_d_shape_v1",
        exact_steps_multiplier=10,
    )
    for idx, params in enumerate(_profile_seed_trials(profile="high_amp_guarded_d_shape_v1")):
        tokens = _build_realtime_tokens(
            base_cfg=base,
            params=params,
            output_json=Path(f"out_{idx}.json"),
            run_tag=f"trial_guarded_seed_{idx}",
        )
        assert "--checkpoint-controller-progress-early-stop-site-error-mean-max" not in tokens
        args = build_realtime_parser().parse_args(tokens)
        assert args.checkpoint_controller_exact_forecast_guardrail_mode == "d_shape_barrier_v1"


def test_realtime_parser_and_controller_config_accept_single_surface_commit_law_flag() -> None:
    parser = build_realtime_parser()
    args = parser.parse_args(
        [
            "--artifact-json",
            "artifact.json",
            "--output-json",
            "out.json",
            "--checkpoint-controller-exact-v1-single-surface-commit-law",
        ]
    )

    cfg = build_controller_config(args)

    assert args.checkpoint_controller_exact_v1_single_surface_commit_law is True
    assert cfg.exact_v1_single_surface_commit_law is True


@pytest.mark.parametrize(
    "profile",
    [
        "high_amp_guarded_d_shape_live_v1",
        "high_amp_guarded_d_shape_turnwindow_v1",
        "high_amp_guarded_d_shape_expanded_v1",
        "high_amp_guarded_d_shape_site3_prefsite_autobaseline_v1",
        "high_amp_guarded_fidelity_first_site3_prefsite_autobaseline_v1",
        "high_amp_guarded_site3_full_surface_v2",
    ],
)
def test_live_guarded_profile_defaults_use_density_first_feasible_ranking(profile: str) -> None:
    args = realtime_optuna.build_parser().parse_args(
        [
            "--artifact-json",
            "artifact.json",
            "--study-profile",
            profile,
        ]
    )
    weights = realtime_optuna._build_objective_weights(args)
    gates = realtime_optuna._build_validity_gates(args)
    assert weights.pair_mae_over_exact_span == pytest.approx(1.0)
    assert weights.epsilon_osc_pair == pytest.approx(0.75)
    assert weights.dominant_peak_abs_omega_error == pytest.approx(0.5)
    assert weights.mean_abs_energy_total_error == pytest.approx(0.0)
    assert weights.fidelity_defect == pytest.approx(0.0)
    assert weights.append_count == pytest.approx(0.02)
    assert gates.max_mean_abs_energy_total_error == pytest.approx(0.25)
    assert gates.max_final_abs_energy_total_error == pytest.approx(0.25)
    assert gates.min_fidelity_exact == pytest.approx(0.90)
    assert gates.max_mean_total_occupation_abs_error == pytest.approx(0.05)


def test_live_guarded_high_amp_baseline_round_trip_through_current_parser() -> None:
    base = BaseRunConfig(
        artifact_json=Path("artifact.json"),
        study_profile="high_amp_guarded_d_shape_live_v1",
        enable_drive=True,
        drive_A=1.5,
        drive_omega=1.2,
        drive_tbar=4.0,
        exact_steps_multiplier=10,
    )
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=_baseline_trial_params(profile="high_amp_guarded_d_shape_live_v1"),
        output_json=Path("out.json"),
        run_tag="trial_guarded_live",
    )
    assert "--checkpoint-controller-progress-observable-window" not in tokens
    assert "--checkpoint-controller-exact-v1-below-floor-energy-safe-turn-escape" not in tokens
    assert "--no-checkpoint-controller-exact-v1-d-shape-pre-turn-shadow-bridge" in tokens
    args = build_realtime_parser().parse_args(tokens)
    assert args.checkpoint_controller_prune_mode == "exact_local_v1"
    assert args.checkpoint_controller_exact_forecast_guardrail_mode == "d_shape_barrier_v1"
    assert args.checkpoint_controller_exact_forecast_drive_harmonic_weight == pytest.approx(0.0)
    assert args.checkpoint_controller_exact_v1_repeat_reopen_mode == "sign_reversal_window"
    assert args.checkpoint_controller_exact_v1_d_shape_turn_window_abs_activation == pytest.approx(0.04)
    assert (
        args.checkpoint_controller_exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold
        == 0
    )
    assert args.checkpoint_controller_exact_v1_d_shape_pre_turn_shadow_bridge is False
    assert args.checkpoint_controller_exact_v1_density_first_target_gain_floor == pytest.approx(0.02)
    assert args.checkpoint_controller_exact_v1_below_floor_probe_target_gain_floor == pytest.approx(0.03)
    assert args.checkpoint_controller_exact_v1_sign_lag_window_activation == pytest.approx(0.1)
    assert args.checkpoint_controller_exact_v1_sign_lag_window_target_gain_floor == pytest.approx(0.005)


def test_live_guarded_profile_seed_trials_cover_harmonic_seed() -> None:
    seeds = _profile_seed_trials(profile="high_amp_guarded_d_shape_live_v1")
    assert any(seed.drive_harmonic_weight == pytest.approx(0.25) for seed in seeds)


def test_turnwindow_profile_seed_trials_cover_bridge_and_window_variants() -> None:
    seeds = _profile_seed_trials(profile="high_amp_guarded_d_shape_turnwindow_v1")
    assert any(seed.d_shape_pre_turn_shadow_bridge is True for seed in seeds)
    assert any(seed.d_shape_turn_window_abs_activation == pytest.approx(0.02) for seed in seeds)
    assert any(seed.d_shape_outside_turn_below_floor_probe_stall_threshold == 2 for seed in seeds)


def test_expanded_profile_seed_trials_cover_structural_variants() -> None:
    seeds = _profile_seed_trials(profile="high_amp_guarded_d_shape_expanded_v1")
    assert any(seed.baseline_proposal_mode == "anticipatory_drive_basis_v1" for seed in seeds)
    assert any(seed.primary_density_target_mode == "pair_difference" for seed in seeds)
    assert any(seed.confirm_score_mode == "exact_gain_ratio" for seed in seeds)
    assert any(seed.include_tangent_secant_proposal is False for seed in seeds)
    assert any(seed.regularization_lambda == pytest.approx(1.0e-6) for seed in seeds)


def test_full_surface_profile_seed_trials_cover_guardrail_and_commit_variants() -> None:
    seeds = _profile_seed_trials(profile=_FULL_SURFACE_PROFILE)
    assert any(seed.guardrail_mode == "d_shape_barrier_v1" for seed in seeds)
    assert any(seed.guardrail_mode == "fidelity_first_barrier_v1" for seed in seeds)
    assert any(seed.single_surface_commit_law is True for seed in seeds)
    assert any(seed.include_tangent_secant_proposal is False for seed in seeds)


@pytest.mark.parametrize(("profile", "guardrail_mode"), _PREFSITE_AUTOBASELINE_PROFILES)
def test_prefsite_autobaseline_profile_seed_trials_cover_geometry_variants(
    profile: str,
    guardrail_mode: str,
) -> None:
    seeds = _profile_seed_trials(profile=profile)
    assert all(seed.guardrail_mode == guardrail_mode for seed in seeds)
    assert any(seed.confirm_score_mode == "exact_gain_ratio" for seed in seeds)
    assert any(seed.shortlist_size == 6 for seed in seeds)
    assert any(seed.regularization_lambda == pytest.approx(1.0e-6) for seed in seeds)
    assert any(seed.trust_radius == pytest.approx(1.15) for seed in seeds)


def test_turnwindow_baseline_round_trip_through_current_parser() -> None:
    base = BaseRunConfig(
        artifact_json=Path("artifact.json"),
        study_profile="high_amp_guarded_d_shape_turnwindow_v1",
        enable_drive=True,
        drive_A=1.5,
        drive_omega=1.2,
        drive_tbar=4.0,
        exact_steps_multiplier=10,
    )
    params = _baseline_trial_params(profile="high_amp_guarded_d_shape_turnwindow_v1")
    params = TrialParams(
        **{
            **params.__dict__,
            "d_shape_turn_window_abs_activation": 0.03,
            "d_shape_outside_turn_below_floor_probe_stall_threshold": 2,
            "d_shape_pre_turn_shadow_bridge": True,
            "drive_harmonic_weight": 0.1,
        }
    )
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=params,
        output_json=Path("out.json"),
        run_tag="trial_guarded_turnwindow",
    )
    assert "--checkpoint-controller-exact-v1-d-shape-pre-turn-shadow-bridge" in tokens
    args = build_realtime_parser().parse_args(tokens)
    assert args.checkpoint_controller_exact_forecast_guardrail_mode == "d_shape_barrier_v1"
    assert args.checkpoint_controller_exact_forecast_drive_harmonic_weight == pytest.approx(0.1)
    assert args.checkpoint_controller_exact_v1_d_shape_turn_window_abs_activation == pytest.approx(0.03)
    assert (
        args.checkpoint_controller_exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold
        == 2
    )
    assert args.checkpoint_controller_exact_v1_d_shape_pre_turn_shadow_bridge is True


def test_expanded_baseline_round_trip_through_current_parser() -> None:
    base = BaseRunConfig(
        artifact_json=Path("artifact.json"),
        study_profile="high_amp_guarded_d_shape_expanded_v1",
        enable_drive=True,
        drive_A=1.5,
        drive_omega=1.2,
        drive_tbar=4.0,
        exact_steps_multiplier=10,
    )
    params = _baseline_trial_params(profile="high_amp_guarded_d_shape_expanded_v1")
    params = TrialParams(
        **{
            **params.__dict__,
            "include_tangent_secant_proposal": False,
            "baseline_proposal_mode": "anticipatory_drive_basis_v1",
            "primary_density_target_mode": "pair_difference",
            "append_margin_abs": 1.0e-5,
            "shortlist_size": 6,
            "shortlist_fraction": 0.25,
            "active_window_size": 4,
            "max_probe_positions": 6,
            "regularization_lambda": 1.0e-6,
            "candidate_regularization_lambda": 1.0e-10,
            "pinv_rcond": 1.0e-8,
            "compile_penalty_weight": 0.0,
            "measurement_penalty_weight": 0.05,
            "directional_penalty_weight": 0.03,
            "confirm_score_mode": "exact_gain_ratio",
            "confirm_compress_fraction": 0.25,
            "confirm_compress_min_modes": 2,
            "confirm_compress_max_modes": 12,
            "d_shape_turn_window_abs_activation": 0.02,
            "d_shape_outside_turn_below_floor_probe_stall_threshold": 2,
            "d_shape_pre_turn_shadow_bridge": True,
        }
    )
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=params,
        output_json=Path("out.json"),
        run_tag="trial_guarded_expanded",
    )
    assert "--no-checkpoint-controller-exact-forecast-include-tangent-secant-proposal" in tokens
    assert "--checkpoint-controller-exact-v1-d-shape-pre-turn-shadow-bridge" in tokens
    args = build_realtime_parser().parse_args(tokens)
    assert args.checkpoint_controller_exact_forecast_baseline_proposal_mode == "anticipatory_drive_basis_v1"
    assert args.checkpoint_controller_exact_forecast_primary_density_target_mode == "pair_difference"
    assert args.checkpoint_controller_append_margin_abs == pytest.approx(1.0e-5)
    assert args.checkpoint_controller_shortlist_size == 6
    assert args.checkpoint_controller_shortlist_fraction == pytest.approx(0.25)
    assert args.checkpoint_controller_active_window_size == 4
    assert args.checkpoint_controller_max_probe_positions == 6
    assert args.checkpoint_controller_regularization_lambda == pytest.approx(1.0e-6)
    assert args.checkpoint_controller_candidate_regularization_lambda == pytest.approx(1.0e-10)
    assert args.checkpoint_controller_pinv_rcond == pytest.approx(1.0e-8)
    assert args.checkpoint_controller_compile_penalty_weight == pytest.approx(0.0)
    assert args.checkpoint_controller_measurement_penalty_weight == pytest.approx(0.05)
    assert args.checkpoint_controller_directional_penalty_weight == pytest.approx(0.03)
    assert args.checkpoint_controller_confirm_score_mode == "exact_gain_ratio"
    assert args.checkpoint_controller_confirm_compress_fraction == pytest.approx(0.25)
    assert args.checkpoint_controller_confirm_compress_min_modes == 2
    assert args.checkpoint_controller_confirm_compress_max_modes == 12
    assert args.checkpoint_controller_exact_forecast_include_tangent_secant_proposal is False
    assert args.checkpoint_controller_exact_v1_d_shape_turn_window_abs_activation == pytest.approx(0.02)
    assert (
        args.checkpoint_controller_exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold
        == 2
    )
    assert args.checkpoint_controller_exact_v1_d_shape_pre_turn_shadow_bridge is True


def test_full_surface_baseline_round_trip_through_current_parser() -> None:
    base = BaseRunConfig(
        artifact_json=Path("artifact.json"),
        study_profile=_FULL_SURFACE_PROFILE,
        enable_drive=True,
        drive_A=1.5,
        drive_omega=1.2,
        drive_tbar=4.0,
        exact_steps_multiplier=10,
    )
    params = _baseline_trial_params(profile=_FULL_SURFACE_PROFILE)
    params = TrialParams(
        **{
            **params.__dict__,
            "oracle_selection_policy": "measured_topk_oracle_energy",
            "candidate_step_scales": "0.1,0.2,0.3,0.4,0.5",
            "baseline_blend_weights": "-0.125,0.0,0.125,0.25",
            "baseline_gain_scales": "0.5,0.75,1.0,1.25",
            "horizon_spec": "4|4.0,2.0,1.0,1.0",
            "tracking_staggered_weight": 2.0,
            "guardrail_mode": "fidelity_first_barrier_v1",
            "single_surface_commit_law": True,
            "prune_mode": "exact_local_v1",
            "prune_miss_threshold": 0.05,
            "prune_loss_threshold": 0.02,
            "prune_theta_block_tol": 0.1,
            "prune_state_jump_l2_tol": 0.1,
            "prune_safe_miss_increase_tol": 0.02,
        }
    )
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=params,
        output_json=Path("out.json"),
        run_tag="trial_full_surface",
    )
    assert "--checkpoint-controller-exact-v1-single-surface-commit-law" in tokens
    args = build_realtime_parser().parse_args(tokens)
    assert args.checkpoint_controller_oracle_selection_policy == "measured_topk_oracle_energy"
    assert args.checkpoint_controller_candidate_step_scales == "0.1,0.2,0.3,0.4,0.5"
    assert (
        args.checkpoint_controller_exact_forecast_baseline_blend_weights
        == "-0.125,0.0,0.125,0.25"
    )
    assert args.checkpoint_controller_exact_forecast_baseline_gain_scales == "0.5,0.75,1.0,1.25"
    assert args.checkpoint_controller_exact_forecast_horizon_steps == 4
    assert args.checkpoint_controller_exact_forecast_horizon_weights == "4.0,2.0,1.0,1.0"
    assert args.checkpoint_controller_exact_forecast_tracking_staggered_error_weight == pytest.approx(2.0)
    assert args.checkpoint_controller_exact_forecast_guardrail_mode == "fidelity_first_barrier_v1"
    assert args.checkpoint_controller_exact_v1_single_surface_commit_law is True
    assert args.checkpoint_controller_prune_miss_threshold == pytest.approx(0.05)
    assert args.checkpoint_controller_prune_loss_threshold == pytest.approx(0.02)
    assert args.checkpoint_controller_prune_theta_block_tol == pytest.approx(0.1)
    assert args.checkpoint_controller_prune_state_jump_l2_tol == pytest.approx(0.1)
    assert args.checkpoint_controller_prune_safe_miss_increase_tol == pytest.approx(0.02)


@pytest.mark.parametrize(("profile", "guardrail_mode"), _PREFSITE_AUTOBASELINE_PROFILES)
def test_prefsite_autobaseline_baseline_round_trip_through_current_parser(
    profile: str,
    guardrail_mode: str,
) -> None:
    base = BaseRunConfig(
        artifact_json=Path("artifact.json"),
        study_profile=profile,
        enable_drive=True,
        drive_A=1.5,
        drive_omega=1.2,
        drive_tbar=4.0,
        exact_steps_multiplier=10,
    )
    params = _baseline_trial_params(profile=profile)
    params = TrialParams(
        **{
            **params.__dict__,
            "append_margin_abs": 1.0e-5,
            "shortlist_size": 6,
            "shortlist_fraction": 0.25,
            "active_window_size": 4,
            "max_probe_positions": 6,
            "regularization_lambda": 1.0e-6,
            "candidate_regularization_lambda": 1.0e-10,
            "pinv_rcond": 1.0e-8,
            "compile_penalty_weight": 0.0,
            "measurement_penalty_weight": 0.05,
            "directional_penalty_weight": 0.03,
            "confirm_score_mode": "exact_gain_ratio",
            "trust_radius": 1.15,
            "signed_energy_lead_limit": 4.0,
            "fidelity_loss_tol": 0.015,
            "abs_energy_error_increase_tol": 0.03,
            "total_occupation_error_increase_tol": 0.015,
        }
    )
    tokens = _build_realtime_tokens(
        base_cfg=base,
        params=params,
        output_json=Path("out.json"),
        run_tag="trial_prefsite_autobaseline",
    )
    assert "--checkpoint-controller-progress-observable-window" not in tokens
    assert "--checkpoint-controller-progress-early-stop-min-checkpoint" not in tokens
    args = build_realtime_parser().parse_args(tokens)
    assert args.checkpoint_controller_exact_forecast_baseline_proposal_mode == "norm_locked_blend_v1"
    assert args.checkpoint_controller_exact_forecast_primary_density_target_mode == "pair_difference"
    assert args.checkpoint_controller_append_margin_abs == pytest.approx(1.0e-5)
    assert args.checkpoint_controller_shortlist_size == 6
    assert args.checkpoint_controller_shortlist_fraction == pytest.approx(0.25)
    assert args.checkpoint_controller_active_window_size == 4
    assert args.checkpoint_controller_max_probe_positions == 6
    assert args.checkpoint_controller_regularization_lambda == pytest.approx(1.0e-6)
    assert args.checkpoint_controller_candidate_regularization_lambda == pytest.approx(1.0e-10)
    assert args.checkpoint_controller_pinv_rcond == pytest.approx(1.0e-8)
    assert args.checkpoint_controller_compile_penalty_weight == pytest.approx(0.0)
    assert args.checkpoint_controller_measurement_penalty_weight == pytest.approx(0.05)
    assert args.checkpoint_controller_directional_penalty_weight == pytest.approx(0.03)
    assert args.checkpoint_controller_confirm_score_mode == "exact_gain_ratio"
    assert args.checkpoint_controller_exact_forecast_guardrail_mode == guardrail_mode
    assert args.checkpoint_controller_exact_forecast_include_tangent_secant_proposal is True
    assert args.checkpoint_controller_exact_v1_d_shape_turn_window_abs_activation == pytest.approx(0.04)
    assert (
        args.checkpoint_controller_exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold
        == 7
    )
    assert args.checkpoint_controller_exact_v1_d_shape_pre_turn_shadow_bridge is True
    assert args.checkpoint_controller_progress_early_stop_site_error_mean_max is None


def test_full_surface_suggest_trial_params_skips_dead_dependency_knobs() -> None:
    class _FakeTrial:
        def __init__(self, picks: dict[str, object]):
            self.picks = picks
            self.calls: list[str] = []

        def suggest_categorical(self, name: str, choices: list[object]) -> object:
            self.calls.append(name)
            value = self.picks[name]
            assert value in choices
            return value

    space = _search_space(profile=_FULL_SURFACE_PROFILE)
    picks = {key: list(values)[0] for key, values in space.items()}
    picks["include_tangent_secant_proposal"] = False
    picks["confirm_score_mode"] = "exact_gain_ratio"
    picks["prune_mode"] = "off"
    trial = _FakeTrial(picks)
    params = realtime_optuna._suggest_trial_params(trial, profile=_FULL_SURFACE_PROFILE)
    baseline = _baseline_trial_params(profile=_FULL_SURFACE_PROFILE)
    assert params.include_tangent_secant_proposal is False
    assert params.confirm_score_mode == "exact_gain_ratio"
    assert params.prune_mode == "off"
    assert params.trust_radius == pytest.approx(baseline.trust_radius)
    assert params.signed_energy_lead_limit == pytest.approx(baseline.signed_energy_lead_limit)
    assert params.confirm_compress_fraction == pytest.approx(baseline.confirm_compress_fraction)
    assert params.prune_miss_threshold == pytest.approx(baseline.prune_miss_threshold)
    assert "trust_radius" not in trial.calls
    assert "signed_energy_lead_limit" not in trial.calls
    assert "confirm_compress_fraction" not in trial.calls
    assert "confirm_compress_min_modes" not in trial.calls
    assert "confirm_compress_max_modes" not in trial.calls
    assert "prune_miss_threshold" not in trial.calls
    assert "prune_loss_threshold" not in trial.calls
    assert "prune_theta_block_tol" not in trial.calls
    assert "prune_state_jump_l2_tol" not in trial.calls
    assert "prune_safe_miss_increase_tol" not in trial.calls


def test_trial_metrics_from_payload_tracks_guarded_admissions(tmp_path: Path) -> None:
    payload = _synthetic_payload()
    payload["summary"] = {
        **dict(payload["summary"]),
        "status": "completed",
        "early_stop_reason": "none",
        "exact_forecast_veto_count": 2,
        "decision_override_count": 1,
        "append_count": 1,
    }
    trajectory = list(payload["trajectory"])
    trajectory[0]["exact_v1_selection_reason"] = "below_near_miss_floor"
    trajectory[0]["proposed_action_kind"] = "stay"
    trajectory[0]["decision_override_reason"] = None
    trajectory[1]["exact_v1_selection_reason"] = "d_shape_barrier_protected_horizon"
    trajectory[1]["proposed_action_kind"] = "append_candidate"
    trajectory[1]["decision_override_reason"] = None
    trajectory[2]["exact_v1_selection_reason"] = "no_tracking_win_vs_stay"
    trajectory[2]["proposed_action_kind"] = "append_candidate"
    trajectory[2]["decision_override_reason"] = "exact_forecast_d_shape_barrier_veto"
    payload["trajectory"] = trajectory
    output_json = tmp_path / "guarded_result.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _, metrics = _trial_metrics_from_payload(
        payload=payload,
        output_json=output_json,
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            pair="auto",
        ),
    )
    assert metrics["summary_status"] == "completed"
    assert metrics["early_stop_reason"] is None
    assert metrics["exact_forecast_veto_count"] == 2
    assert metrics["decision_override_count"] == 1
    assert metrics["protected_horizon_selection_count"] == 1
    assert metrics["proposed_append_count"] == 2
    assert metrics["append_commit_rate"] == pytest.approx(0.5)
    assert metrics["guardrail_d_shape_override_count"] == 1
    assert metrics["selection_reason_counts"]["d_shape_barrier_protected_horizon"] == 1
    assert metrics["decision_override_reason_counts"]["exact_forecast_d_shape_barrier_veto"] == 1


def test_best_by_metric_skips_completed_trials_that_fail_full_horizon_gate() -> None:
    early = realtime_optuna.TrialObservation(
        trial_number=1,
        params={},
        objective=0.1,
        status="completed",
        metrics={"pair_mae_over_exact_span": 0.1, "full_horizon_gate_passed": False},
        invalid_reasons=[],
    )
    complete = realtime_optuna.TrialObservation(
        trial_number=2,
        params={},
        objective=0.2,
        status="completed",
        metrics={"pair_mae_over_exact_span": 0.2, "full_horizon_gate_passed": True},
        invalid_reasons=[],
    )

    best = realtime_optuna._best_by_metric([early, complete], key="pair_mae_over_exact_span")

    assert best is not None
    assert best["trial_number"] == 2


def test_write_json_serializes_paths_and_nonfinite(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    _write_json(path, {"artifact": Path("artifact.json"), "nan_metric": float("nan")})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "artifact.json"
    assert payload["nan_metric"] is None


def test_evaluate_trial_converts_system_exit_into_failed_observation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Parser:
        def parse_args(self, _tokens: list[str]) -> object:
            raise SystemExit(2)

    monkeypatch.setattr(realtime_optuna, "build_realtime_parser", lambda: _Parser())
    obs = _evaluate_trial(
        trial_number=0,
        params=TrialParams(),
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json"), disable_drive=True),
        objective_weights=ObjectiveWeights(),
        validity_gates=ValidityGates(),
        output_dir=tmp_path,
    )
    assert obs.status == "failed"
    assert obs.invalid_reasons == ["system_exit"]
    assert obs.error is not None and "SystemExit" in obs.error


def test_evaluate_trial_skips_spectra_pdf_by_default_and_user_attrs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(*, args: object, params: TrialParams) -> dict[str, object]:
        del params
        payload = _synthetic_payload()
        output_json = Path(str(getattr(args, "output_json")))
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    monkeypatch.setattr(
        realtime_optuna,
        "_run_realtime_from_args_with_optuna_overrides",
        _fake_run,
    )

    obs = _evaluate_trial(
        trial_number=0,
        params=TrialParams(),
        base_cfg=BaseRunConfig(artifact_json=Path("artifact.json"), disable_drive=True),
        objective_weights=ObjectiveWeights(),
        validity_gates=ValidityGates(),
        output_dir=tmp_path,
    )

    assert obs.spectra_json is not None
    assert obs.spectra_pdf is None
    assert Path(obs.spectra_json).exists()
    spectra_pdf = tmp_path / "trials" / "trial_0000" / "spectra.pdf"
    assert not spectra_pdf.exists()
    attrs = realtime_optuna._observation_to_user_attrs(obs)
    assert attrs["spectra_pdf"] is None
    summary_path = tmp_path / "trials" / "trial_0000" / "trial_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["spectra_pdf"] is None


def test_evaluate_trial_writes_spectra_pdf_when_opted_in_and_user_attrs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(*, args: object, params: TrialParams) -> dict[str, object]:
        del params
        payload = _synthetic_payload()
        output_json = Path(str(getattr(args, "output_json")))
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    monkeypatch.setattr(
        realtime_optuna,
        "_run_realtime_from_args_with_optuna_overrides",
        _fake_run,
    )

    obs = _evaluate_trial(
        trial_number=0,
        params=TrialParams(),
        base_cfg=BaseRunConfig(
            artifact_json=Path("artifact.json"),
            disable_drive=True,
            skip_spectra_pdf=False,
        ),
        objective_weights=ObjectiveWeights(),
        validity_gates=ValidityGates(),
        output_dir=tmp_path,
    )

    assert obs.spectra_json is not None
    assert obs.spectra_pdf is not None
    assert Path(obs.spectra_json).exists()
    spectra_pdf = Path(obs.spectra_pdf)
    assert spectra_pdf.exists()
    assert spectra_pdf.read_bytes().startswith(b"%PDF")
    attrs = realtime_optuna._observation_to_user_attrs(obs)
    assert attrs["spectra_pdf"] == str(spectra_pdf)
    summary_path = tmp_path / "trials" / "trial_0000" / "trial_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["spectra_pdf"] == str(spectra_pdf)
