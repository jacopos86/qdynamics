from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.tables import generic_dynamics_ablation_matrix as ablate
from pipelines.time_dynamics.runners import hh_from_adapt_artifact as realtime_mod
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import DynamicsBenchmarkCase


def _case(tmp_path: Path, *, metadata: dict | None = None) -> DynamicsBenchmarkCase:
    artifact = tmp_path / "seed.json"
    artifact.write_text("{}", encoding="utf-8")
    return DynamicsBenchmarkCase(
        case_id="unit_hubbard",
        family="hubbard",
        table_class="fermionic_lattice",
        artifact_json=str(artifact),
        t_final=0.2,
        num_times=3,
        metadata=dict(metadata or {}),
    )


def _hh_case(tmp_path: Path, *, amplitude: float = 0.2) -> DynamicsBenchmarkCase:
    artifact = tmp_path / "hh_seed.json"
    artifact.write_text("{}", encoding="utf-8")
    return DynamicsBenchmarkCase(
        case_id=f"table1_hh_snake_A{str(amplitude).replace('.', 'p')}_t8_dt321_seedtracks_v2",
        family="hh",
        table_class="hubbard_holstein",
        artifact_json=str(artifact),
        t_final=8.0,
        num_times=321,
        generator_family="full_meta",
        fallback_family="full_meta",
        append_pool_family="full_meta",
        tuning_class="hybrid",
        metadata={
            "enable_drive": True,
            "drive": {
                "enable_drive": True,
                "A": amplitude,
                "omega": 1.0,
                "tbar": 1.0,
                "pattern": "staggered",
                "time_sampling": "midpoint",
            },
            "seed_lock": {"seed_track": "snake"},
        },
    )


def _strict_payload(
    *,
    append_count: int = 1,
    prune_count: int = 1,
    integrator_policy: str = "rk4",
    euler_count: int = 0,
    rk4_count: int = 1,
    append_enabled: bool | None = None,
    prune_mode: str | None = None,
    forced_euler: bool = False,
) -> dict:
    controller_config = {
        "mode": "observable_v1",
        "append_enabled": append_enabled,
        "prune_mode": prune_mode,
        "integrator_policy": integrator_policy,
    }
    controller_config = {k: v for k, v in controller_config.items() if v is not None}
    ledger_row = {
        "decision_backend": "ideal_observable",
        "decision_noise_mode": "ideal",
        "controller_exact_input_mode": "off",
        "decision_data_flow": "ideal_observable_estimator",
        "uses_reference_for_decision": False,
        "uses_future_exact_forecast_for_decision": False,
        "uses_statevector_as_ideal_observable_estimator": True,
        "strict_measurement_oracle_certified": True,
        "action_kind": "stay",
        "integrator_used": "rk4" if rk4_count else ("euler" if euler_count else "none"),
    }
    if forced_euler:
        ledger_row["integrator_forced_policy"] = "euler"
    return {
        "summary": {
            "mode": "observable_v1",
            "controller_exact_input_mode": "off",
            "diagnostic_exact_reference_mode": "benchmark_exact",
            "decision_data_flow": "ideal_observable_estimator",
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "uses_statevector_as_ideal_observable_estimator": True,
            "strict_measurement_oracle_certified": True,
            "append_count": int(append_count),
            "prune_count": int(prune_count),
            "integrator_euler_count": int(euler_count),
            "integrator_rk4_count": int(rk4_count),
            "integrator_policy": str(integrator_policy),
            "max_abs_site_occupations_error": 0.12,
            "shots_total": 256,
        },
        "controller_config": controller_config,
        "runtime_contract": {
            "controller_exact_input_mode": "off",
            "diagnostic_exact_reference_mode": "benchmark_exact",
            "decision_data_flow": "ideal_observable_estimator",
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
        },
        "compile_audit": {
            "selected_backend": {
                "compiled_count_2q": 5,
                "compiled_depth_2q": 6,
                "compiled_depth": 9,
                "compiled_size": 11,
                "backend_name": "unit_backend",
            }
        },
        "trajectory": [
            {
                "time": 0.0,
                "energy_total": 1.0,
                "energy_total_exact": 1.0,
                "abs_energy_total_error": 0.0,
                "abs_primary_density_error": 0.0,
                "fidelity_exact": 1.0,
                "runtime_parameter_count": 2,
            },
            {
                "time": 0.2,
                "energy_total": 1.1,
                "energy_total_exact": 1.0,
                "abs_energy_total_error": 0.1,
                "abs_primary_density_error": 0.03,
                "fidelity_exact": 0.98,
                "runtime_parameter_count": 3,
            },
        ],
        "ledger": [ledger_row],
    }


def test_time_dynamics_parser_builds_no_append_controller_config(tmp_path: Path) -> None:
    args = realtime_mod.build_parser().parse_args(
        [
            "--artifact-json",
            str(tmp_path / "seed.json"),
            "--output-json",
            str(tmp_path / "out.json"),
            "--no-checkpoint-controller-append-enabled",
            "--checkpoint-controller-confirm-score-mode",
            "exact_gain_ratio",
            "--checkpoint-controller-exact-input-mode",
            "off",
        ]
    )
    cfg = realtime_mod.build_controller_config(args)

    assert args.checkpoint_controller_append_enabled is False
    assert args.checkpoint_controller_integrator_policy == "auto_euler_rk4"
    assert args.checkpoint_controller_reference_mode == "off"
    assert cfg.append_enabled is False
    assert cfg.confirm_score_mode == "exact_gain_ratio"
    assert cfg.reference_mode == "off"


def test_strict_ablation_guard_raises_on_exact_decision_leakage() -> None:
    variant = ablate.get_controller_ablation_variant("full_controller")
    payload = _strict_payload()
    payload["summary"]["uses_reference_for_decision"] = True

    with pytest.raises(ValueError, match="uses_reference_for_decision"):
        ablate.validate_ablation_decision_data_flow(payload=payload, variant=variant)


def test_strict_ablation_guard_inspects_runtime_contract_metadata() -> None:
    variant = ablate.get_controller_ablation_variant("full_controller")
    payload = _strict_payload()
    payload["runtime_contract"]["decision_data_flow"] = "exact_assisted_controller"

    with pytest.raises(ValueError, match="runtime_contract.decision_data_flow"):
        ablate.validate_ablation_decision_data_flow(payload=payload, variant=variant)


def test_auto_euler_v2_runtime_guard_rejects_euler_when_gate_failed() -> None:
    variant = ablate.get_controller_ablation_variant("hh_recovery_s5_auto_no_append_no_prune")
    payload = _strict_payload(
        append_count=0,
        prune_count=0,
        integrator_policy="auto_euler_rk4",
        euler_count=1,
        rk4_count=0,
        append_enabled=False,
        prune_mode="off",
    )
    row = payload["ledger"][0]
    row.update(
        {
            "integrator_used": "euler",
            "integrator_auto_policy_schema": "auto_euler_rk4_policy_v2",
            "integrator_geometry_gate_pass": False,
            "integrator_euler_error_pass": True,
            "integrator_condition_pass": True,
            "integrator_rho_miss_pass": True,
            "integrator_euler_time_gate_pass": True,
            "integrator_euler_observable_gate_pass": True,
            "integrator_auto_admit_euler": False,
            "integrator_euler_blockers": ["geometry"],
        }
    )

    validation = ablate.validate_ablation_variant_runtime(payload=payload, variant=variant)

    assert validation["passed"] is False
    assert any("auto_euler_v2_gate_violation" in item for item in validation["violations"])



def test_fixed_scaffold_guard_rejects_controller_exact_input_mode() -> None:
    variant = ablate.get_controller_ablation_variant("fixed_scaffold")
    payload = {
        "summary": {
            "mode": "off",
            "controller_exact_input_mode": "benchmark_exact",
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
        },
        "trajectory": [],
    }

    with pytest.raises(ValueError, match="controller_exact_input_mode=benchmark_exact"):
        ablate.validate_ablation_decision_data_flow(payload=payload, variant=variant)


def test_fixed_scaffold_guard_allows_diagnostic_reference_block() -> None:
    variant = ablate.get_controller_ablation_variant("fixed_scaffold")
    payload = {
        "summary": {
            "mode": "off",
            "controller_exact_input_mode": "off",
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
        },
        "reference": {
            "reference_enabled": True,
            "reference_mode": "benchmark_exact",
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
        },
        "trajectory": [],
    }

    contract = ablate.validate_ablation_decision_data_flow(payload=payload, variant=variant)

    assert contract["passed"] is True


def test_strict_ablation_guard_allows_reporting_only_benchmark_exact_reference_block() -> None:
    variant = ablate.get_controller_ablation_variant("full_controller")
    payload = _strict_payload()
    payload["reference"] = {
        "reference_enabled": True,
        "reference_mode": "benchmark_exact",
        "uses_reference_for_decision": False,
        "uses_future_exact_forecast_for_decision": False,
    }

    contract = ablate.validate_ablation_decision_data_flow(payload=payload, variant=variant)

    assert contract["passed"] is True



def test_strict_ablation_guard_ignores_serialized_exact_forecast_defaults() -> None:
    variant = ablate.get_controller_ablation_variant("full_controller")
    payload = _strict_payload()
    payload["summary"].update(
        {
            "exact_forecast_baseline_proposal_mode": "norm_locked_blend_v1",
            "exact_forecast_tracking_horizon_steps": 1,
            "exact_forecast_tracking_fidelity_defect_weight": 1.0,
            "exact_forecast_density_slope_weight": 1.0,
            "exact_forecast_guardrail_mode": "off",
            "exact_forecast_veto_count": 0,
        }
    )

    contract = ablate.validate_ablation_decision_data_flow(payload=payload, variant=variant)

    assert contract["passed"] is True


def test_strict_ablation_guard_ignores_inactive_exact_forecast_veto_counter() -> None:
    variant = ablate.get_controller_ablation_variant("full_controller")
    payload = _strict_payload()
    payload["summary"].update(
        {
            "exact_forecast_guardrail_mode": "off",
            "exact_forecast_veto_count": 1,
            "uses_future_exact_forecast_for_decision": False,
        }
    )

    contract = ablate.validate_ablation_decision_data_flow(payload=payload, variant=variant)

    assert contract["passed"] is True


def test_strict_ablation_guard_rejects_active_exact_forecast_guardrail() -> None:
    variant = ablate.get_controller_ablation_variant("full_controller")
    payload = _strict_payload()
    payload["summary"]["exact_forecast_guardrail_mode"] = "dual_metric_v1"

    with pytest.raises(ValueError, match="exact_forecast_guardrail_mode"):
        ablate.validate_ablation_decision_data_flow(payload=payload, variant=variant)

def test_no_append_runtime_knob_guard_fails_if_append_fired(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_from_args(args):
        return _strict_payload(append_count=1, prune_count=0)

    monkeypatch.setattr(ablate.realtime, "run_from_args", fake_run_from_args)

    with pytest.raises(ValueError, match="knob guard failed: append_count=1"):
        ablate.run_generic_controller_ablation_row(
            case=_case(tmp_path),
            variant_id="no_append",
            output_dir=tmp_path / "bad_no_append",
        )


def test_no_append_variant_uses_append_disabled_cli_and_diagnostic_exact_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_run_from_args(args):
        captured["args"] = args
        return _strict_payload(append_count=0, prune_count=1)

    monkeypatch.setattr(ablate.realtime, "run_from_args", fake_run_from_args)

    row = ablate.run_generic_controller_ablation_row(
        case=_case(tmp_path),
        variant_id="no_append",
        output_dir=tmp_path / "no_append",
    ).to_dict()

    args = captured["args"]
    assert args.checkpoint_controller_append_enabled is False
    assert args.checkpoint_controller_integrator_policy == "rk4"
    assert args.checkpoint_controller_reference_mode == "off"
    assert args.diagnostic_exact_reference_mode == "benchmark_exact"
    assert args.checkpoint_controller_strict_qpu_faithful is True
    assert row["schema"] == "dynamics_benchmark_row_v1"
    assert row["status"] == "completed"
    assert row["algorithm_id"] == "dyn_controller_no_append"
    assert row["qpu_faithful"] is True
    assert row["exact_assisted"] is False
    assert row["diagnostic"] is True
    assert row["metrics"]["append_count"] == 0
    assert row["metrics"]["strict_decision_contract_passed"] is True
    assert row["provenance"]["append_enabled"] is False
    assert row["provenance"]["controller_exact_input_mode"] == "off"
    assert row["provenance"]["diagnostic_exact_reference_mode"] == "benchmark_exact"
    assert row["provenance"]["exact_references_reporting_only"] is True
    assert row["provenance"]["tuning_granularity"] == "coarse_hamiltonian_class"
    assert row["provenance"]["source_table_class"] == "fermionic_lattice"
    assert row["provenance"]["tuning_class"] == "fermionic"
    assert row["provenance"]["controller_settings_id"] == row["provenance"]["settings_id"]
    assert row["provenance"]["static_scaffold_scope"] == "benchmark_point"
    assert row["provenance"]["class_tuned_result_locked"] is False
    assert row["table_fields"]["table_status_label"] == "no append"
    raw = json.loads((tmp_path / "no_append" / "raw_payload.json").read_text(encoding="utf-8"))
    assert raw["row_contract"] == {
        "qpu_faithful": True,
        "exact_assisted": False,
        "diagnostic": True,
        "paper_promotion_eligible": True,
    }
    assert raw["parameter_manifest"]["tuning_provenance"]["settings_id"] == row["provenance"]["settings_id"]


def test_hh_recovery_variants_are_explicit_only() -> None:
    default_ids = {variant.variant_id for variant in ablate.default_controller_ablation_variants()}
    all_ids = {variant.variant_id for variant in ablate.controller_ablation_variants()}

    assert "hh_recovery_s1_rk4_no_append_no_prune" in all_ids
    assert "hh_recovery_s1_rk4_no_append_no_prune" not in default_ids
    assert "hh_recovery_s5_auto_no_append_no_prune" in all_ids
    assert "hh_recovery_s5_auto_no_append_no_prune" not in default_ids
    assert "full_controller" in default_ids


def test_hh_recovery_stage1_forces_rk4_and_exact_free_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_run_from_args(args):
        captured["args"] = args
        return _strict_payload(
            append_count=0,
            prune_count=0,
            integrator_policy="rk4",
            euler_count=0,
            rk4_count=2,
            append_enabled=False,
            prune_mode="off",
        )

    monkeypatch.setattr(ablate.realtime, "run_from_args", fake_run_from_args)

    row = ablate.run_generic_controller_ablation_row(
        case=_hh_case(tmp_path),
        variant_id="hh_recovery_s1_rk4_no_append_no_prune",
        output_dir=tmp_path / "hh_s1",
    ).to_dict()

    args = captured["args"]
    assert args.checkpoint_controller_mode == "observable_v1"
    assert args.checkpoint_controller_append_enabled is False
    assert args.checkpoint_controller_prune_mode == "off"
    assert args.checkpoint_controller_integrator_policy == "rk4"
    assert args.checkpoint_controller_reference_mode == "off"
    assert args.diagnostic_exact_reference_mode == "benchmark_exact"
    assert args.checkpoint_controller_strict_qpu_faithful is True
    assert row["algorithm_id"] == "dyn_hh_recovery_s1_rk4_no_append_no_prune"
    assert row["metrics"]["diagnostic_ladder_id"] == "hh_recovery_ladder_v1"
    assert row["metrics"]["diagnostic_ladder_stage"] == 1
    assert row["metrics"]["paper_promotion_eligible"] is False
    assert row["provenance"]["stage_knob_source"] == "hh_recovery_ladder_v1_explicit_stage_override"
    assert row["provenance"]["class_lock_role"] == "baseline_hybrid_policy_only"


def test_hh_recovery_stage4_auto_integrator_uses_hh_observable_guardrails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_run_from_args(args):
        captured["args"] = args
        return _strict_payload(
            append_count=1,
            prune_count=1,
            integrator_policy="auto_euler_rk4",
            euler_count=1,
            rk4_count=1,
            append_enabled=True,
            prune_mode="schur_projected_shadow_v1",
        )

    monkeypatch.setattr(ablate.realtime, "run_from_args", fake_run_from_args)

    row = ablate.run_generic_controller_ablation_row(
        case=_hh_case(tmp_path),
        variant_id="hh_recovery_s4_auto_append_prune",
        output_dir=tmp_path / "hh_s4",
    ).to_dict()

    args = captured["args"]
    assert args.checkpoint_controller_append_enabled is True
    assert args.checkpoint_controller_prune_mode == "schur_projected_shadow_v1"
    assert args.checkpoint_controller_integrator_policy == "auto_euler_rk4"
    assert args.checkpoint_controller_integrator_euler_site_span_max == pytest.approx(1.0e-2)
    assert args.checkpoint_controller_integrator_euler_primary_density_span_max == pytest.approx(2.0e-2)
    assert args.checkpoint_controller_integrator_euler_energy_span_max == pytest.approx(2.0e-3)
    assert row["algorithm_id"] == "dyn_hh_recovery_s4_auto_append_prune"
    assert row["metrics"]["diagnostic_ladder_stage"] == 4
    assert row["provenance"]["hh_auto_euler_guardrail_applied"] is True



def test_hh_recovery_stage5_is_auto_integrator_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_run_from_args(args):
        captured["args"] = args
        return _strict_payload(
            append_count=0,
            prune_count=0,
            integrator_policy="auto_euler_rk4",
            euler_count=1,
            rk4_count=1,
            append_enabled=False,
            prune_mode="off",
        )

    monkeypatch.setattr(ablate.realtime, "run_from_args", fake_run_from_args)

    row = ablate.run_generic_controller_ablation_row(
        case=_hh_case(tmp_path),
        variant_id="hh_recovery_s5_auto_no_append_no_prune",
        output_dir=tmp_path / "hh_s5",
    ).to_dict()

    args = captured["args"]
    assert args.checkpoint_controller_append_enabled is False
    assert args.checkpoint_controller_prune_mode == "off"
    assert args.checkpoint_controller_integrator_policy == "auto_euler_rk4"
    assert args.checkpoint_controller_integrator_euler_site_span_max == pytest.approx(1.0e-2)
    assert args.checkpoint_controller_integrator_euler_primary_density_span_max == pytest.approx(2.0e-2)
    assert args.checkpoint_controller_integrator_euler_energy_span_max == pytest.approx(2.0e-3)
    assert row["algorithm_id"] == "dyn_hh_recovery_s5_auto_no_append_no_prune"
    assert row["metrics"]["diagnostic_ladder_stage"] == 5
    assert row["metrics"]["append_count"] == 0
    assert row["metrics"]["prune_count"] == 0
    assert row["metrics"]["integrator_euler_count"] == 1
    assert row["provenance"]["hh_auto_euler_guardrail_applied"] is True
    assert row["provenance"]["hh_auto_euler_guardrail_id"] == "hh_auto_euler_observable_guardrails_v1"
    assert row["provenance"]["controller_decisions_modified"] is True


def test_hh_recovery_fixed_rk4_guard_rejects_euler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_from_args(args):
        return _strict_payload(
            append_count=0,
            prune_count=0,
            integrator_policy="rk4",
            euler_count=1,
            rk4_count=1,
            append_enabled=False,
            prune_mode="off",
        )

    monkeypatch.setattr(ablate.realtime, "run_from_args", fake_run_from_args)

    with pytest.raises(ValueError, match="integrator_euler_count=1"):
        ablate.run_generic_controller_ablation_row(
            case=_hh_case(tmp_path),
            variant_id="hh_recovery_s1_rk4_no_append_no_prune",
            output_dir=tmp_path / "hh_s1_bad_euler",
        )


def test_hh_recovery_variants_reject_non_hh_cases(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires HH L2"):
        ablate.run_generic_controller_ablation_row(
            case=_case(tmp_path),
            variant_id="hh_recovery_s1_rk4_no_append_no_prune",
            output_dir=tmp_path / "bad_family",
        )


def test_ablation_matrix_retains_failed_variant_and_table_skip_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_from_args(args):
        if args.checkpoint_controller_append_enabled is False:
            raise RuntimeError("unit no-append failure")
        return _strict_payload(append_count=2, prune_count=1)

    monkeypatch.setattr(ablate.realtime, "run_from_args", fake_run_from_args)

    result = ablate.run_generic_controller_ablation_matrix(
        case=_case(tmp_path),
        output_dir=tmp_path / "matrix",
        variants=("full_controller", "no_append"),
    )

    assert result["schema"] == "generic_controller_ablation_matrix_v1"
    assert result["status_counts"] == {"completed": 1, "failed": 1}
    by_algorithm = {row["algorithm_id"]: row for row in result["rows"]}
    assert by_algorithm["dyn_controller_full"]["status"] == "completed"
    assert by_algorithm["dyn_controller_no_append"]["status"] == "failed"
    assert "unit no-append failure" in by_algorithm["dyn_controller_no_append"]["reason"]
    table = result["tables"]["tab:dyn_ablation_matrix"]
    assert table["status_counts"] == {"completed": 1, "failed": 1}
    failed_table_row = next(row for row in table["rows"] if row["variant_id"] == "no_append")
    assert failed_table_row["paired_with_full"] is True
    assert failed_table_row["paired_full_status"] == "completed"
    assert failed_table_row["delta_mean_abs_energy_total_error_disabled_minus_full"] is None
    assert (tmp_path / "matrix" / "tab_dyn_ablation_matrix.json").exists()


def test_ablation_runner_inherits_locked_full_policy_before_no_append_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings_manifest = tmp_path / "class_settings.json"
    settings_manifest.write_text(
        json.dumps(
            {
                "schema": "dynamics_class_settings_lock_manifest_v1",
                "lock_status": "locked",
                "settings": [
                    {
                        "tuning_class": "fermionic",
                        "algorithm_id": "dyn_controller_full",
                        "settings_kind": "controller",
                        "variant_id": "full_controller",
                        "settings_source": "unit_class_optuna_v1",
                        "class_tuned_result_locked": True,
                        "settings_payload": {
                            "miss_threshold": 0.37,
                            "gain_ratio_threshold": 0.013,
                            "append_enabled": True,
                            "prune_mode": "schur_projected_shadow_v1",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_from_args(args):
        captured["args"] = args
        return _strict_payload(append_count=0, prune_count=1)

    monkeypatch.setattr(ablate.realtime, "run_from_args", fake_run_from_args)
    case = ablate.with_class_settings_lock_manifest(
        _case(tmp_path),
        manifest_path=settings_manifest,
        require_locked=True,
    )

    row = ablate.run_generic_controller_ablation_row(
        case=case,
        variant_id="no_append",
        output_dir=tmp_path / "no_append_locked",
    ).to_dict()

    assert captured["args"].checkpoint_controller_miss_threshold == pytest.approx(0.37)
    assert captured["args"].checkpoint_controller_gain_ratio_threshold == pytest.approx(0.013)
    assert captured["args"].checkpoint_controller_append_enabled is False
    assert row["provenance"]["ablation_base_policy_algorithm_id"] == "dyn_controller_full"
    assert row["provenance"]["ablation_base_policy_variant_id"] == "full_controller"
    assert row["provenance"]["class_tuned_result_locked"] is True
    assert row["provenance"]["settings_source"] == "unit_class_optuna_v1"


def test_ablation_runner_applies_class_locked_controller_cli_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings_manifest = tmp_path / "class_settings.json"
    settings_manifest.write_text(
        json.dumps(
            {
                "schema": "dynamics_class_settings_lock_manifest_v1",
                "lock_status": "locked",
                "settings": [
                    {
                        "tuning_class": "fermionic",
                        "algorithm_id": "dyn_controller_full",
                        "settings_kind": "controller",
                        "variant_id": "full_controller",
                        "settings_source": "unit_class_optuna_v1",
                        "class_tuned_result_locked": True,
                        "settings_payload": {
                            "miss_threshold": 0.37,
                            "gain_ratio_threshold": 0.013,
                            "append_enabled": True,
                            "prune_mode": "schur_projected_shadow_v1",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_from_args(args):
        captured["args"] = args
        return _strict_payload(append_count=1, prune_count=1)

    monkeypatch.setattr(ablate.realtime, "run_from_args", fake_run_from_args)
    case = ablate.with_class_settings_lock_manifest(
        _case(
            tmp_path,
            metadata={
                "enable_drive": True,
                "drive": {
                    "A": 0.6,
                    "omega": 1.7,
                    "tbar": 2.0,
                    "phi": 0.25,
                    "pattern": "staggered",
                    "time_sampling": "midpoint",
                },
            },
        ),
        manifest_path=settings_manifest,
        require_locked=True,
    )

    row = ablate.run_generic_controller_ablation_row(
        case=case,
        variant_id="full_controller",
        output_dir=tmp_path / "full_locked",
    ).to_dict()

    assert captured["args"].enable_drive is True
    assert captured["args"].drive_A == pytest.approx(0.6)
    assert captured["args"].drive_omega == pytest.approx(1.7)
    assert captured["args"].drive_tbar == pytest.approx(2.0)
    assert captured["args"].drive_phi == pytest.approx(0.25)
    assert captured["args"].drive_pattern == "staggered"
    assert captured["args"].checkpoint_controller_miss_threshold == pytest.approx(0.37)
    assert captured["args"].checkpoint_controller_gain_ratio_threshold == pytest.approx(0.013)
    assert captured["args"].checkpoint_controller_append_enabled is True
    assert captured["args"].checkpoint_controller_integrator_policy == "rk4"
    assert row["provenance"]["class_tuned_result_locked"] is True
    assert row["provenance"]["settings_source"] == "unit_class_optuna_v1"
    assert row["provenance"]["same_seed_comparator_group_id"] == "unit_hubbard"
