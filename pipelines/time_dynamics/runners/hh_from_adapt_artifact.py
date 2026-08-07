"""Archaic checkpoint-controller runner.

This route is retained for legacy tests and artifact replay only. It is not the
active Paper-II AP-McLachlan solve-repair implementation; use
``ap_append_from_adapt_artifact.py`` and ``ap_mclachlan/fixed_step.py`` for the
Paper-II candidate-set state-space repair policy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.noise_oracle_runtime import (
    OracleConfig,
    assess_oracle_execution_capability,
    normalize_oracle_execution_request,
)
from pipelines.static_adapt.builders.problem_setup import _resolve_exact_energy_from_payload
from pipelines.hardcoded.hh_fixed_manifold_mclachlan import (
    FixedManifoldRunSpec,
    load_run_context,
)
from pipelines.time_dynamics.legacy.checkpoint_controller import (
    ControllerDriveConfig,
    RealtimeCheckpointController,
)
from pipelines.time_dynamics.legacy.checkpoint_exact_audit import (
    RealtimeExactAuditHelper,
    build_exact_audit_helper_for_controller,
    run_controller_with_exact_audit,
)
from pipelines.time_dynamics.legacy.checkpoint_types import (
    HIGH_MISS_NO_ADMIT_POLICY_DEFAULT,
    RealtimeCheckpointConfig,
    decision_data_flow_fields,
    normalize_high_miss_no_admit_policy,
    normalize_reference_mode,
    normalize_realtime_controller_mode,
)
from pipelines.time_dynamics.legacy.checkpoint_measurement import (
    validate_controller_oracle_base_config,
)
from pipelines.time_dynamics.legacy.checkpoint_compile_audit import (
    build_compile_audit_config_from_args,
    compile_audit_summary_mirrors,
    run_final_scaffold_compile_audit,
    run_prune_event_compile_audit,
)
from pipelines.time_dynamics.legacy.checkpoint_route_defaults import (
    DRIVE_A_DEFAULT,
    DRIVE_DEFAULTS_SOURCE,
    DRIVE_OMEGA_DEFAULT,
    DRIVE_PATTERN_DEFAULT,
    DRIVE_PHI_DEFAULT,
    DRIVE_T0_DEFAULT,
    DRIVE_TBAR_DEFAULT,
    DRIVE_TIME_SAMPLING_DEFAULT,
    ENABLE_DRIVE_DEFAULT,
    EXACT_STEPS_MULTIPLIER_DEFAULT,
    NUM_TIMES_DEFAULT,
    ROUTE_AUTHORITY,
    ROUTE_LABEL,
    ROUTE_VERSION,
    T_FINAL_DEFAULT,
)
from pipelines.time_dynamics.legacy.checkpoint_route_policy import (
    STRICT_QPU_FLAG_LABEL,
    strict_qpu_faithful_requested,
    validate_realtime_route_request,
)
from pipelines.time_dynamics.adapters.hamiltonian import adapter_for_family_key
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


def _parse_float_tuple(raw: str | None) -> tuple[float, ...]:
    if raw is None:
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(float(chunk.strip()) for chunk in text.split(",") if chunk.strip())


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    return value


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return float(out) if np.isfinite(out) else None


def _controller_energy_from_row(row: Mapping[str, Any]) -> float | None:
    for key in ("energy_total_controller", "energy_total"):
        out = _finite_float_or_none(row.get(key))
        if out is not None:
            return float(out)
    return None


def _exact_energy_target_from_loaded(loaded: Any) -> tuple[float | None, str | None]:
    payload_candidates = (
        getattr(loaded, "payload", None),
        getattr(getattr(loaded, "replay_context", None), "payload_in", None),
    )
    for payload in payload_candidates:
        if isinstance(payload, Mapping):
            exact_energy = _resolve_exact_energy_from_payload(payload)
            if exact_energy is not None:
                return float(exact_energy), "artifact_payload"
    return None, None


def _problem_family_from_loaded(
    *,
    loaded: Any,
    replay_context: Any,
    explicit_problem_family: str | None,
) -> str:
    if explicit_problem_family not in {None, ""}:
        return str(explicit_problem_family)
    runtime_input = getattr(loaded, "runtime_input", None)
    resolved_problem = (
        None if runtime_input is None else getattr(runtime_input, "resolved_problem", None)
    )
    family_key = None if resolved_problem is None else getattr(resolved_problem, "family_key", None)
    if family_key not in {None, ""}:
        return str(family_key)
    payload_in = getattr(replay_context, "payload_in", None)
    if isinstance(payload_in, Mapping):
        settings = payload_in.get("settings", {})
        if isinstance(settings, Mapping):
            problem = settings.get("problem", None)
            if problem not in {None, ""}:
                return str(problem)
    return "hh"


def annotate_ed_ground_energy_target(
    payload: dict[str, Any],
    *,
    exact_energy: float | None,
    source: str | None,
    drive_enabled: bool,
) -> None:
    """Add physical ED-ground energy target metrics alongside seed-trajectory metrics.

    The existing benchmark_exact columns compare against exact time evolution of the
    ADAPT/scaffold seed.  These target columns compare controller energy to the
    exact diagonalized static ground energy carried by the source artifact, so a
    bad seed cannot look physically perfect merely by being tracked exactly.
    """

    exact = _finite_float_or_none(exact_energy)
    enabled = exact is not None
    target_payload = {
        "enabled": bool(enabled),
        "kind": "static_hamiltonian_ed_ground_energy",
        "exact_energy": exact,
        "source": source,
        "note": (
            "Energy target is the static ED ground-state energy from the seed artifact. "
            "Existing abs_energy_total_error remains the exact propagation error from the ADAPT/scaffold seed."
        ),
    }
    if bool(drive_enabled):
        target_payload["drive_scope"] = (
            "static_ground_energy_only; driven ED-ground trajectory target is not yet emitted"
        )
    if not enabled:
        return
    reference = payload.setdefault("reference", {})
    if isinstance(reference, dict):
        reference.setdefault("ed_ground_energy_target", dict(target_payload))

    summary = payload.setdefault("summary", {})
    if not isinstance(summary, dict):
        return
    if summary.get("final_abs_energy_total_error_to_ed_ground") is not None:
        return
    summary["ed_ground_energy_target_enabled"] = True
    summary["ed_ground_energy_target_kind"] = str(target_payload["kind"])
    summary["ed_ground_energy_target_exact_energy"] = exact
    summary["ed_ground_energy_target_source"] = source
    if bool(drive_enabled):
        summary["ed_ground_energy_target_drive_scope"] = target_payload["drive_scope"]

    errors: list[float] = []
    rows = payload.get("trajectory", [])
    if isinstance(rows, list):
        for raw_row in rows:
            if not isinstance(raw_row, dict):
                continue
            energy = _controller_energy_from_row(raw_row)
            if energy is None:
                continue
            raw_row["energy_total_ed_ground"] = float(exact)
            raw_row["abs_energy_total_error_to_ed_ground"] = float(
                abs(float(energy) - float(exact))
            )
            errors.append(float(raw_row["abs_energy_total_error_to_ed_ground"]))

    if errors:
        summary["initial_abs_energy_total_error_to_ed_ground"] = float(errors[0])
        summary["final_abs_energy_total_error_to_ed_ground"] = float(errors[-1])
        summary["mean_abs_energy_total_error_to_ed_ground"] = float(np.mean(errors))
        summary["max_abs_energy_total_error_to_ed_ground"] = float(np.max(errors))
    else:
        summary["ed_ground_energy_target_missing_reason"] = "trajectory_energy_unavailable"


def _parse_string_tuple(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(chunk.strip() for chunk in text.split(",") if chunk.strip())


def _oracle_noise_mode_from_args(args: argparse.Namespace) -> str:
    raw_mode = getattr(args, "checkpoint_controller_noise_mode", None)
    if raw_mode not in {None, ""}:
        return str(raw_mode).strip().lower()
    if strict_qpu_faithful_requested(args):
        return "ideal"
    return "backend_scheduled" if bool(getattr(args, "use_fake_backend", False)) else "runtime"


def _default_backend_name_for_mode(mode: str) -> str:
    return "FakeMarrakesh" if str(mode).strip().lower() == "backend_scheduled" else "ibm_marrakesh"


def _build_oracle_mitigation_config(args: argparse.Namespace) -> dict[str, Any]:
    readout_strategy = getattr(args, "final_noise_audit_local_readout_strategy", None)
    readout_strategy = (
        None if readout_strategy in {None, "", "none"} else str(readout_strategy)
    )
    gate_twirling = bool(getattr(args, "final_noise_audit_local_gate_twirling", False))
    dd_sequence = getattr(args, "final_noise_audit_dd_sequence", None)
    dd_sequence = None if dd_sequence in {None, "", "none"} else str(dd_sequence)
    zne_scales = _parse_float_tuple(getattr(args, "final_noise_audit_zne_scales", None))
    zne_extrapolator = _parse_string_tuple(
        getattr(args, "final_noise_audit_zne_extrapolator", None)
    )
    if zne_scales:
        mode = "zne"
    elif dd_sequence is not None:
        mode = "dd"
    elif readout_strategy is not None:
        mode = "readout"
    else:
        mode = "none"
    payload: dict[str, Any] = {"mode": str(mode)}
    if readout_strategy is not None:
        payload["local_readout_strategy"] = str(readout_strategy)
    if gate_twirling:
        payload["local_gate_twirling"] = True
        payload["local_gate_twirling_scope"] = "2q_only"
    if dd_sequence is not None:
        payload["dd_sequence"] = str(dd_sequence)
    if zne_scales:
        payload["zne_scales"] = [float(x) for x in zne_scales]
    if zne_extrapolator:
        payload["zne_extrapolator"] = [str(x) for x in zne_extrapolator]
    return payload


def build_oracle_config(args: argparse.Namespace) -> OracleConfig | None:
    if normalize_realtime_controller_mode(
        getattr(args, "checkpoint_controller_mode", "off")
    ) != "oracle_v1":
        return None
    noise_mode = _oracle_noise_mode_from_args(args)
    strict_qpu_hh = strict_qpu_faithful_requested(args)
    if strict_qpu_hh and str(noise_mode) not in {"ideal", "shots"}:
        raise ValueError(
            f"{STRICT_QPU_FLAG_LABEL} requires oracle noise_mode=ideal or shots."
        )
    if strict_qpu_hh and bool(getattr(args, "use_fake_backend", False)):
        raise ValueError(f"{STRICT_QPU_FLAG_LABEL} forbids --use-fake-backend.")
    value_noise_model = str(
        getattr(args, "checkpoint_controller_value_noise_model", "off")
    ).strip().lower() or "off"
    value_noise_std = float(getattr(args, "checkpoint_controller_value_noise_std", 0.0))
    value_noise_seed = getattr(args, "checkpoint_controller_value_noise_seed", None)
    if strict_qpu_hh and value_noise_model != "off":
        raise ValueError(
            f"{STRICT_QPU_FLAG_LABEL} forbids artificial checkpoint-controller value noise."
        )
    requested_backend = getattr(args, "backend_name", None)
    backend_name = (
        _default_backend_name_for_mode(str(noise_mode))
        if requested_backend in {None, ""}
        else str(requested_backend)
    )
    if strict_qpu_hh:
        backend_name = None
    runtime_profile = getattr(args, "checkpoint_controller_runtime_profile", None)
    if runtime_profile in {None, ""}:
        runtime_profile = getattr(args, "final_noise_audit_runtime_profile", None)
    if runtime_profile in {None, ""}:
        runtime_profile = "legacy_runtime_v0"
    runtime_raw_profile = getattr(args, "checkpoint_controller_runtime_raw_profile", None)
    if runtime_raw_profile in {None, ""}:
        runtime_raw_profile = (
            "raw_sampler_twirled_v1" if str(noise_mode) == "runtime" else "legacy_runtime_v0"
        )
    runtime_session = getattr(args, "checkpoint_controller_runtime_session_policy", None)
    if runtime_session in {None, ""}:
        runtime_session = getattr(args, "final_noise_audit_runtime_session_policy", None)
    if runtime_session in {None, ""}:
        runtime_session = "prefer_session"
    raw_transport = getattr(args, "checkpoint_controller_raw_transport", "auto")
    if raw_transport in {None, "", "auto"} and str(noise_mode) == "runtime":
        raw_transport = "sampler_v2"
    config = OracleConfig(
        noise_mode=str(noise_mode),
        shots=int(getattr(args, "shots")),
        seed=int(getattr(args, "seed")),
        seed_transpiler=int(getattr(args, "seed_transpiler")),
        transpile_optimization_level=int(getattr(args, "transpile_optimization_level")),
        oracle_repeats=int(getattr(args, "oracle_repeats")),
        oracle_aggregate=str(getattr(args, "oracle_aggregate")),
        backend_name=(None if backend_name in {None, ""} else str(backend_name)),
        use_fake_backend=bool(getattr(args, "use_fake_backend", False)),
        allow_aer_fallback=(
            False if strict_qpu_hh else bool(getattr(args, "allow_aer_fallback", True))
        ),
        mitigation=(
            {"mode": "none"}
            if strict_qpu_hh
            else dict(_build_oracle_mitigation_config(args))
        ),
        symmetry_mitigation={"mode": "off"},
        runtime_profile={"name": str(runtime_profile)},
        runtime_raw_profile={"name": str(runtime_raw_profile)},
        runtime_session={"mode": str(runtime_session)},
        raw_transport=("auto" if strict_qpu_hh else str(raw_transport)),
        raw_store_memory=(
            False
            if strict_qpu_hh
            else bool(getattr(args, "checkpoint_controller_raw_store_memory", False))
        ),
        raw_artifact_path=(
            None
            if strict_qpu_hh
            or getattr(args, "checkpoint_controller_raw_artifact_path", None) in {None, ""}
            else str(getattr(args, "checkpoint_controller_raw_artifact_path"))
        ),
        value_noise_model=str(value_noise_model),
        value_noise_std=float(value_noise_std),
        value_noise_seed=value_noise_seed,
    )
    validate_controller_oracle_base_config(config)
    return config


def build_controller_config(args: argparse.Namespace) -> RealtimeCheckpointConfig:
    mode = normalize_realtime_controller_mode(
        getattr(args, "checkpoint_controller_mode", "off")
    )
    reference_mode = normalize_reference_mode(
        getattr(args, "checkpoint_controller_reference_mode", "off")
    )
    primary_density_weight = getattr(
        args,
        "checkpoint_controller_exact_forecast_tracking_primary_density_error_weight",
        None,
    )
    if primary_density_weight is None:
        primary_density_weight = args.checkpoint_controller_exact_forecast_tracking_staggered_error_weight
    cfg = RealtimeCheckpointConfig(
        mode=str(mode),
        reference_mode=str(reference_mode),
        oracle_selection_policy=str(args.checkpoint_controller_oracle_selection_policy),
        miss_threshold=float(args.checkpoint_controller_miss_threshold),
        high_miss_no_admit_policy=normalize_high_miss_no_admit_policy(
            args.checkpoint_controller_high_miss_no_admit_policy
        ),
        repair_retry_max_attempts=int(args.checkpoint_controller_repair_retry_max_attempts),
        repair_retry_escalation_mode=str(args.checkpoint_controller_repair_retry_escalation_mode),
        repair_retry_admission_policy=str(args.checkpoint_controller_repair_retry_admission_policy),
        repair_retry_rescue_min_gain_ratio=float(
            args.checkpoint_controller_repair_retry_rescue_min_gain_ratio
        ),
        repair_retry_rescue_attempt=str(args.checkpoint_controller_repair_retry_rescue_attempt),
        miss_abs_threshold=float(args.checkpoint_controller_miss_abs_threshold),
        miss_persistence_window=int(args.checkpoint_controller_miss_persistence_window),
        miss_persistence_count=int(args.checkpoint_controller_miss_persistence_count),
        integrator_policy=str(args.checkpoint_controller_integrator_policy),
        integrator_columnarity_threshold=float(
            args.checkpoint_controller_integrator_columnarity_threshold
        ),
        integrator_curvature_threshold=float(
            args.checkpoint_controller_integrator_curvature_threshold
        ),
        integrator_euler_fs_error_threshold=float(
            args.checkpoint_controller_integrator_euler_fs_error_threshold
        ),
        integrator_condition_max=float(args.checkpoint_controller_integrator_condition_max),
        integrator_euler_min_time_fraction=float(
            args.checkpoint_controller_integrator_euler_min_time_fraction
        ),
        integrator_euler_observable_window=int(
            args.checkpoint_controller_integrator_euler_observable_window
        ),
        integrator_euler_site_span_max=args.checkpoint_controller_integrator_euler_site_span_max,
        integrator_euler_primary_density_span_max=args.checkpoint_controller_integrator_euler_primary_density_span_max,
        integrator_euler_energy_span_max=args.checkpoint_controller_integrator_euler_energy_span_max,
        gain_ratio_threshold=float(args.checkpoint_controller_gain_ratio_threshold),
        append_margin_abs=float(args.checkpoint_controller_append_margin_abs),
        append_enabled=bool(args.checkpoint_controller_append_enabled),
        append_no_harm_guard_enabled=bool(args.checkpoint_controller_append_no_harm_guard_enabled),
        append_no_harm_condition_ratio_cap=float(args.checkpoint_controller_append_no_harm_condition_ratio_cap),
        append_no_harm_displacement_ratio_cap=float(args.checkpoint_controller_append_no_harm_displacement_ratio_cap),
        append_no_harm_condition_abs_floor=float(args.checkpoint_controller_append_no_harm_condition_abs_floor),
        append_no_harm_kink_min_step_gain_delta=float(args.checkpoint_controller_append_no_harm_kink_min_step_gain_delta),
        append_no_harm_kink_max_condition_ratio=float(args.checkpoint_controller_append_no_harm_kink_max_condition_ratio),
        append_no_harm_kink_max_displacement_ratio=float(args.checkpoint_controller_append_no_harm_kink_max_displacement_ratio),
        append_no_harm_rho_only_min_step_gain_delta=float(args.checkpoint_controller_append_no_harm_rho_only_min_step_gain_delta),
        append_no_harm_rho_only_condition_ratio_cap=float(args.checkpoint_controller_append_no_harm_rho_only_condition_ratio_cap),
        append_no_harm_rho_only_step_residual_ratio_cap=float(args.checkpoint_controller_append_no_harm_rho_only_step_residual_ratio_cap),
        append_no_harm_rho_only_displacement_ratio_cap=float(args.checkpoint_controller_append_no_harm_rho_only_displacement_ratio_cap),
        confirm_score_mode=str(args.checkpoint_controller_confirm_score_mode),
        prune_mode=str(args.checkpoint_controller_prune_mode),
        prune_miss_threshold=float(args.checkpoint_controller_prune_miss_threshold),
        prune_loss_threshold=float(args.checkpoint_controller_prune_loss_threshold),
        prune_theta_block_tol=float(args.checkpoint_controller_prune_theta_block_tol),
        prune_state_jump_l2_tol=float(args.checkpoint_controller_prune_state_jump_l2_tol),
        prune_safe_miss_increase_tol=float(
            args.checkpoint_controller_prune_safe_miss_increase_tol
        ),
        prune_no_harm_guard_enabled=bool(
            args.checkpoint_controller_prune_no_harm_guard_enabled
        ),
        prune_no_harm_score_increase_tol=float(
            args.checkpoint_controller_prune_no_harm_score_increase_tol
        ),
        prune_no_harm_step_residual_ratio_increase_tol=float(
            args.checkpoint_controller_prune_no_harm_step_residual_ratio_increase_tol
        ),
        prune_schur_ladder_local_radius=int(
            args.checkpoint_controller_prune_schur_ladder_local_radius
        ),
        prune_schur_monotonicity_tol=float(
            args.checkpoint_controller_prune_schur_monotonicity_tol
        ),
        prune_loss_norm_epsilon=float(args.checkpoint_controller_prune_loss_norm_epsilon),
        prune_differential_miss_tol=float(
            args.checkpoint_controller_prune_differential_miss_tol
        ),
        prune_high_miss_differential_enabled=bool(
            args.checkpoint_controller_prune_high_miss_differential_enabled
        ),
        prune_projection_mode=str(args.checkpoint_controller_prune_projection_mode),
        prune_projection_rounds=int(args.checkpoint_controller_prune_projection_rounds),
        prune_projection_max_active_runtime=int(
            args.checkpoint_controller_prune_projection_max_active_runtime
        ),
        prune_projection_trust_radius=float(
            args.checkpoint_controller_prune_projection_trust_radius
        ),
        prune_projection_regularization=float(
            args.checkpoint_controller_prune_projection_regularization
        ),
        prune_ray_distance_tol=float(args.checkpoint_controller_prune_ray_distance_tol),
        prune_shadow_enabled=bool(args.checkpoint_controller_prune_shadow_enabled),
        prune_shadow_horizon_steps=int(args.checkpoint_controller_prune_shadow_horizon_steps),
        prune_shadow_score_increase_tol=float(
            args.checkpoint_controller_prune_shadow_score_increase_tol
        ),
        prune_persistence_window=int(args.checkpoint_controller_prune_persistence_window),
        prune_persistence_required=int(args.checkpoint_controller_prune_persistence_required),
        prune_appended_origin_bias_enabled=bool(
            args.checkpoint_controller_prune_appended_origin_bias_enabled
        ),
        prune_appended_origin_target_policy=str(
            args.checkpoint_controller_prune_appended_origin_target_policy
        ),
        prune_appended_origin_grace_steps=int(
            args.checkpoint_controller_prune_appended_origin_grace_steps
        ),
        prune_initial_scaffold_grace_steps=int(
            args.checkpoint_controller_prune_initial_scaffold_grace_steps
        ),
        prune_appended_origin_bias_scale=float(
            args.checkpoint_controller_prune_appended_origin_bias_scale
        ),
        prune_appended_origin_bias_max_factor=float(
            args.checkpoint_controller_prune_appended_origin_bias_max_factor
        ),
        candidate_step_scales=_parse_float_tuple(args.checkpoint_controller_candidate_step_scales),
        exact_forecast_baseline_step_refine_rounds=int(
            args.checkpoint_controller_exact_forecast_baseline_step_refine_rounds
        ),
        exact_forecast_baseline_proposal_mode=str(
            args.checkpoint_controller_exact_forecast_baseline_proposal_mode
        ),
        exact_forecast_baseline_blend_weights=_parse_float_tuple(
            args.checkpoint_controller_exact_forecast_baseline_blend_weights
        ),
        exact_forecast_baseline_gain_scales=_parse_float_tuple(
            args.checkpoint_controller_exact_forecast_baseline_gain_scales
        ),
        exact_forecast_include_tangent_secant_proposal=bool(
            args.checkpoint_controller_exact_forecast_include_tangent_secant_proposal
        ),
        exact_forecast_tangent_secant_trust_radius=float(
            args.checkpoint_controller_exact_forecast_tangent_secant_trust_radius
        ),
        exact_forecast_tangent_secant_signed_energy_lead_limit=float(
            args.checkpoint_controller_exact_forecast_tangent_secant_signed_energy_lead_limit
        ),
        exact_forecast_tracking_horizon_steps=int(
            args.checkpoint_controller_exact_forecast_horizon_steps
        ),
        exact_forecast_tracking_horizon_weights=_parse_float_tuple(
            args.checkpoint_controller_exact_forecast_horizon_weights
        ),
        exact_forecast_primary_density_target_mode=str(
            args.checkpoint_controller_exact_forecast_primary_density_target_mode
        ),
        exact_forecast_tracking_fidelity_defect_weight=float(
            args.checkpoint_controller_exact_forecast_tracking_fidelity_defect_weight
        ),
        exact_forecast_tracking_primary_density_error_weight=float(primary_density_weight),
        exact_forecast_tracking_staggered_error_weight=float(
            args.checkpoint_controller_exact_forecast_tracking_staggered_error_weight
        ),
        exact_forecast_tracking_doublon_error_weight=float(
            args.checkpoint_controller_exact_forecast_tracking_doublon_error_weight
        ),
        exact_forecast_tracking_site_occupations_error_weight=float(
            args.checkpoint_controller_exact_forecast_tracking_site_occupations_error_weight
        ),
        exact_forecast_tracking_energy_total_error_weight=float(
            args.checkpoint_controller_exact_forecast_tracking_energy_total_error_weight
        ),
        exact_forecast_density_slope_weight=float(
            args.checkpoint_controller_exact_forecast_density_slope_weight
        ),
        exact_forecast_density_curvature_weight=float(
            args.checkpoint_controller_exact_forecast_density_curvature_weight
        ),
        exact_forecast_density_excursion_under_weight=float(
            args.checkpoint_controller_exact_forecast_density_excursion_under_weight
        ),
        exact_forecast_density_excursion_over_weight=float(
            args.checkpoint_controller_exact_forecast_density_excursion_over_weight
        ),
        exact_forecast_density_sign_lag_weight=float(
            args.checkpoint_controller_exact_forecast_density_sign_lag_weight
        ),
        exact_forecast_density_postcross_wrong_sign_weight=float(
            args.checkpoint_controller_exact_forecast_density_postcross_wrong_sign_weight
        ),
        exact_forecast_drive_harmonic_weight=float(
            args.checkpoint_controller_exact_forecast_drive_harmonic_weight
        ),
        exact_forecast_energy_slope_weight=float(
            args.checkpoint_controller_exact_forecast_energy_slope_weight
        ),
        exact_forecast_energy_curvature_weight=float(
            args.checkpoint_controller_exact_forecast_energy_curvature_weight
        ),
        exact_forecast_energy_excursion_under_weight=float(
            args.checkpoint_controller_exact_forecast_energy_excursion_under_weight
        ),
        exact_forecast_energy_excursion_over_weight=float(
            args.checkpoint_controller_exact_forecast_energy_excursion_over_weight
        ),
        exact_forecast_energy_excursion_rel_tolerance=float(
            args.checkpoint_controller_exact_forecast_energy_excursion_rel_tolerance
        ),
        exact_v1_repeat_reopen_mode=str(
            args.checkpoint_controller_exact_v1_repeat_reopen_mode
        ),
        exact_v1_density_first_target_gain_floor=float(
            args.checkpoint_controller_exact_v1_density_first_target_gain_floor
        ),
        exact_v1_below_floor_probe_target_gain_floor=float(
            args.checkpoint_controller_exact_v1_below_floor_probe_target_gain_floor
        ),
        exact_v1_sign_lag_window_activation=float(
            args.checkpoint_controller_exact_v1_sign_lag_window_activation
        ),
        exact_v1_sign_lag_window_target_gain_floor=(
            None
            if args.checkpoint_controller_exact_v1_sign_lag_window_target_gain_floor is None
            else float(args.checkpoint_controller_exact_v1_sign_lag_window_target_gain_floor)
        ),
        exact_v1_postcross_wrong_sign_activation=float(
            args.checkpoint_controller_exact_v1_postcross_wrong_sign_activation
        ),
        exact_v1_postcross_wrong_sign_target_gain_floor=(
            None
            if args.checkpoint_controller_exact_v1_postcross_wrong_sign_target_gain_floor is None
            else float(args.checkpoint_controller_exact_v1_postcross_wrong_sign_target_gain_floor)
        ),
        exact_v1_postcross_compare_diag=bool(
            args.checkpoint_controller_exact_v1_postcross_compare_diag
        ),
        exact_v1_below_floor_energy_safe_turn_escape=bool(
            args.checkpoint_controller_exact_v1_below_floor_energy_safe_turn_escape
        ),
        exact_v1_below_floor_energy_safe_d_shape_escape=bool(
            args.checkpoint_controller_exact_v1_below_floor_energy_safe_d_shape_escape
        ),
        exact_v1_d_shape_turn_window_abs_activation=float(
            args.checkpoint_controller_exact_v1_d_shape_turn_window_abs_activation
        ),
        exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold=int(
            args.checkpoint_controller_exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold
        ),
        exact_v1_d_shape_pre_turn_shadow_bridge=bool(
            args.checkpoint_controller_exact_v1_d_shape_pre_turn_shadow_bridge
        ),
        exact_v1_single_surface_commit_law=bool(
            args.checkpoint_controller_exact_v1_single_surface_commit_law
        ),
        exact_forecast_guardrail_mode=str(args.checkpoint_controller_exact_forecast_guardrail_mode),
        exact_forecast_fidelity_loss_tol=float(
            args.checkpoint_controller_exact_forecast_fidelity_loss_tol
        ),
        exact_forecast_abs_energy_error_increase_tol=float(
            args.checkpoint_controller_exact_forecast_abs_energy_error_increase_tol
        ),
        exact_forecast_total_occupation_error_increase_tol=float(
            args.checkpoint_controller_exact_forecast_total_occupation_error_increase_tol
        ),
        shortlist_size=int(args.checkpoint_controller_shortlist_size),
        shortlist_fraction=float(args.checkpoint_controller_shortlist_fraction),
        active_window_size=int(args.checkpoint_controller_active_window_size),
        measurement_active_window_size=int(
            args.checkpoint_controller_measurement_active_window_size
        ),
        max_probe_positions=int(args.checkpoint_controller_max_probe_positions),
        regularization_lambda=float(args.checkpoint_controller_regularization_lambda),
        candidate_regularization_lambda=float(
            args.checkpoint_controller_candidate_regularization_lambda
        ),
        pinv_rcond=float(args.checkpoint_controller_pinv_rcond),
        compile_penalty_weight=float(args.checkpoint_controller_compile_penalty_weight),
        measurement_penalty_weight=float(args.checkpoint_controller_measurement_penalty_weight),
        directional_penalty_weight=float(args.checkpoint_controller_directional_penalty_weight),
        confirm_compress_fraction=float(args.checkpoint_controller_confirm_compress_fraction),
        confirm_compress_min_modes=int(args.checkpoint_controller_confirm_compress_min_modes),
        confirm_compress_max_modes=int(args.checkpoint_controller_confirm_compress_max_modes),
        progress_observable_window=int(args.checkpoint_controller_progress_observable_window),
        progress_early_stop_min_checkpoint=int(
            args.checkpoint_controller_progress_early_stop_min_checkpoint
        ),
        progress_early_stop_site_error_mean_max=(
            None
            if args.checkpoint_controller_progress_early_stop_site_error_mean_max is None
            else float(args.checkpoint_controller_progress_early_stop_site_error_mean_max)
        ),
        progress_early_stop_primary_density_error_mean_max=(
            None
            if args.checkpoint_controller_progress_early_stop_primary_density_error_mean_max is None
            else float(args.checkpoint_controller_progress_early_stop_primary_density_error_mean_max)
        ),
        progress_early_stop_energy_error_mean_max=(
            None
            if args.checkpoint_controller_progress_early_stop_energy_error_mean_max is None
            else float(args.checkpoint_controller_progress_early_stop_energy_error_mean_max)
        ),
        progress_early_stop_site_span_max=(
            None
            if args.checkpoint_controller_progress_early_stop_site_span_max is None
            else float(args.checkpoint_controller_progress_early_stop_site_span_max)
        ),
        progress_early_stop_primary_density_span_max=(
            None
            if args.checkpoint_controller_progress_early_stop_primary_density_span_max is None
            else float(args.checkpoint_controller_progress_early_stop_primary_density_span_max)
        ),
        progress_early_stop_energy_span_max=(
            None
            if args.checkpoint_controller_progress_early_stop_energy_span_max is None
            else float(args.checkpoint_controller_progress_early_stop_energy_span_max)
        ),
    )
    if strict_qpu_faithful_requested(args):
        if str(mode) not in {"observable_v1", "oracle_v1"}:
            raise ValueError(
                f"{STRICT_QPU_FLAG_LABEL} requires --checkpoint-controller-mode "
                "observable_v1 or oracle_v1."
            )
        if str(reference_mode) != "off":
            raise ValueError(
                f"{STRICT_QPU_FLAG_LABEL} requires reference-mode off / controller exact inputs off "
                "(--checkpoint-controller-reference-mode/--checkpoint-controller-exact-input-mode off)."
            )
        if normalize_high_miss_no_admit_policy(
            args.checkpoint_controller_high_miss_no_admit_policy
        ) != HIGH_MISS_NO_ADMIT_POLICY_DEFAULT:
            raise ValueError(
                f"{STRICT_QPU_FLAG_LABEL} requires high-miss/no-admit policy bounded_stay_advance."
            )
        if str(mode) == "oracle_v1" and str(args.checkpoint_controller_prune_mode) != "off":
            raise ValueError(f"{STRICT_QPU_FLAG_LABEL} oracle_v1 forbids prune mode.")
        if int(args.checkpoint_controller_progress_early_stop_min_checkpoint) != 0:
            raise ValueError(
                f"{STRICT_QPU_FLAG_LABEL} forbids progress early-stop configuration."
            )
        if any(
            value is not None
            for value in (
                args.checkpoint_controller_progress_early_stop_site_error_mean_max,
                args.checkpoint_controller_progress_early_stop_primary_density_error_mean_max,
                args.checkpoint_controller_progress_early_stop_energy_error_mean_max,
                args.checkpoint_controller_progress_early_stop_site_span_max,
                args.checkpoint_controller_progress_early_stop_primary_density_span_max,
                args.checkpoint_controller_progress_early_stop_energy_span_max,
            )
        ):
            raise ValueError(
                f"{STRICT_QPU_FLAG_LABEL} forbids progress early-stop thresholds."
            )
        if str(args.checkpoint_controller_exact_forecast_guardrail_mode) != "off":
            raise ValueError(f"{STRICT_QPU_FLAG_LABEL} forbids exact forecast guardrails.")
        if str(args.checkpoint_controller_oracle_selection_policy) != "measured_gain_commit_veto":
            raise ValueError(
                f"{STRICT_QPU_FLAG_LABEL} currently requires oracle selection policy measured_gain_commit_veto."
            )
        if bool(args.checkpoint_controller_exact_forecast_include_tangent_secant_proposal):
            raise ValueError(
                f"{STRICT_QPU_FLAG_LABEL} forbids exact tangent/secant proposals."
            )
        strict_replace = {
            "exact_forecast_baseline_step_refine_rounds": 0,
            "exact_forecast_baseline_blend_weights": (),
            "exact_forecast_baseline_gain_scales": (),
            "exact_forecast_include_tangent_secant_proposal": False,
            "exact_forecast_tangent_secant_trust_radius": 0.0,
            "exact_forecast_tangent_secant_signed_energy_lead_limit": 0.0,
            "exact_forecast_tracking_horizon_steps": 1,
            "exact_forecast_tracking_horizon_weights": (),
            "exact_forecast_guardrail_mode": "off",
            "exact_v1_repeat_reopen_mode": "off",
            "exact_v1_postcross_compare_diag": False,
            "exact_v1_below_floor_energy_safe_turn_escape": False,
            "exact_v1_below_floor_energy_safe_d_shape_escape": False,
            "exact_v1_d_shape_pre_turn_shadow_bridge": False,
            "exact_v1_single_surface_commit_law": False,
            "progress_early_stop_min_checkpoint": 0,
            "progress_early_stop_site_error_mean_max": None,
            "progress_early_stop_primary_density_error_mean_max": None,
            "progress_early_stop_energy_error_mean_max": None,
            "progress_early_stop_site_span_max": None,
            "progress_early_stop_primary_density_span_max": None,
            "progress_early_stop_energy_span_max": None,
        }
        if str(mode) == "oracle_v1":
            strict_replace["append_no_harm_guard_enabled"] = False
        return replace(cfg, **strict_replace)
    return cfg


def build_drive_config(
    args: argparse.Namespace,
    *,
    n_sites: int,
    ordering: str,
) -> ControllerDriveConfig | None:
    if not bool(getattr(args, "enable_drive", False)):
        return None
    exact_steps_multiplier = int(args.exact_steps_multiplier)
    if exact_steps_multiplier < 1:
        raise ValueError("--exact-steps-multiplier must be >= 1 when drive is enabled.")
    custom_weights = _parse_float_tuple(args.drive_custom_weights) or None
    if str(args.drive_pattern) == "custom" and custom_weights is None:
        raise ValueError("--drive-pattern custom requires --drive-custom-weights when drive is enabled.")
    return ControllerDriveConfig(
        enabled=True,
        n_sites=int(n_sites),
        ordering=str(ordering),
        drive_A=float(args.drive_A),
        drive_omega=float(args.drive_omega),
        drive_tbar=float(args.drive_tbar),
        drive_phi=float(args.drive_phi),
        drive_pattern=str(args.drive_pattern),
        drive_custom_weights=custom_weights,
        drive_include_identity=bool(args.drive_include_identity),
        drive_time_sampling=str(args.drive_time_sampling),
        drive_t0=float(args.drive_t0),
        exact_steps_multiplier=exact_steps_multiplier,
    )


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _abs_error_or_none(lhs: Any, rhs: Any) -> float | None:
    lhs_f = _finite_or_none(lhs)
    rhs_f = _finite_or_none(rhs)
    if lhs_f is None or rhs_f is None:
        return None
    return float(abs(float(lhs_f) - float(rhs_f)))


def _vector_abs_error_or_none(lhs: Any, rhs: Any) -> list[float] | None:
    if not isinstance(lhs, Sequence) or isinstance(lhs, (str, bytes, bytearray)):
        return None
    if not isinstance(rhs, Sequence) or isinstance(rhs, (str, bytes, bytearray)):
        return None
    lhs_arr = np.asarray(lhs, dtype=float).reshape(-1)
    rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
    if lhs_arr.shape != rhs_arr.shape or lhs_arr.size == 0:
        return None
    return [float(x) for x in np.abs(lhs_arr - rhs_arr).tolist()]


def _vector_error_max_or_none(lhs: Any, rhs: Any) -> float | None:
    err = _vector_abs_error_or_none(lhs, rhs)
    if not err:
        return None
    return float(max(float(x) for x in err))


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = _finite_or_none(row.get(key, None))
        if value is not None:
            out.append(float(value))
    return out


def _summary_stats_for_values(values: Sequence[float], *, prefix: str) -> dict[str, float]:
    arr = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return {}
    return {
        f"initial_{prefix}": float(arr[0]),
        f"final_{prefix}": float(arr[-1]),
        f"mean_{prefix}": float(np.mean(arr)),
        f"max_{prefix}": float(np.max(arr)),
    }


def _diagnostic_exact_step_hmat(
    *,
    controller: RealtimeCheckpointController,
    hmat_static: np.ndarray,
    physical_time: float,
) -> np.ndarray:
    drive_model = getattr(controller, "_drive_model", None)
    if drive_model is not None:
        drive_coeff = float(drive_model.coefficient_at(float(physical_time)))
        if abs(float(drive_coeff)) <= 1.0e-15:
            return np.asarray(hmat_static, dtype=complex)
        drive_hmat = np.asarray(hamiltonian_matrix(drive_model.drive_poly), dtype=complex)
        return np.asarray(np.asarray(hmat_static, dtype=complex) + float(drive_coeff) * drive_hmat, dtype=complex)
    drive_provider = getattr(controller, "_drive_coeff_provider_exyz", None)
    if getattr(controller, "_drive_config", None) is not None and drive_provider is not None:
        from pipelines.hardcoded.hh_fixed_manifold_measured import (
            FixedManifoldMeasuredConfig,
            _build_driven_hamiltonian,
        )

        _h_poly_step, hmat_step, _drive_coeff_map = _build_driven_hamiltonian(
            h_poly_static=controller.h_poly,
            hmat_static=np.asarray(hmat_static, dtype=complex),
            drive_coeff_provider_exyz=drive_provider,
            physical_time=float(physical_time),
            nq=int(getattr(controller, "_num_qubits")),
            geom_cfg=FixedManifoldMeasuredConfig(),
            drive_drop_abs_tol=1.0e-15,
        )
        return np.asarray(hmat_step, dtype=complex)
    return np.asarray(hmat_static, dtype=complex)


def _replace_result_like(result: Any, **changes: Any) -> Any:
    """Return a result with updated fields for dataclass and lightweight test stubs."""

    if is_dataclass(result) and not isinstance(result, type):
        return replace(result, **changes)
    if isinstance(result, SimpleNamespace):
        payload = dict(vars(result))
        payload.update(changes)
        return SimpleNamespace(**payload)
    payload = dict(getattr(result, "__dict__", {}))
    if payload:
        payload.update(changes)
        return SimpleNamespace(**payload)
    raise TypeError(f"cannot replace result fields on {type(result).__name__}")


def _attach_diagnostic_exact_reference(
    *,
    args: argparse.Namespace,
    controller: RealtimeCheckpointController,
    result: Any,
    exact_reference_cache: dict[str, object] | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    """Attach exact benchmark curves as diagnostics without changing controller inputs.

    `checkpoint_controller_reference_mode` remains the controller exact-input guard.
    This helper implements the separate report/artifact side channel: exact ED
    dynamics may be computed for plots and error metrics after the controller run,
    but those values are not available to action selection.
    """

    diagnostic_mode = normalize_reference_mode(
        getattr(args, "diagnostic_exact_reference_mode", "benchmark_exact")
    )
    summary = dict(result.summary)
    trajectory = [dict(row) for row in result.trajectory]
    ledger = [dict(row) for row in result.ledger]
    summary.setdefault("controller_reference_mode", summary.get("reference_mode", "off"))
    summary.setdefault("controller_reference_enabled", bool(summary.get("reference_enabled", False)))
    summary.setdefault(
        "controller_exact_input_mode",
        summary.get("controller_reference_mode", summary.get("reference_mode", "off")),
    )
    summary.setdefault("uses_reference_for_decision", False)
    summary.setdefault("uses_future_exact_forecast_for_decision", False)
    summary["diagnostic_exact_reference_mode"] = str(diagnostic_mode)
    if diagnostic_mode == "off":
        summary["diagnostic_exact_reference_enabled"] = False
        diagnostic_reference = {
            "diagnostic_reference_mode": "off",
            "diagnostic_reference_enabled": False,
            "role": "diagnostic_exact_benchmark",
            "feeds_controller_decisions": False,
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
        }
        return _replace_result_like(
            result, summary=summary, trajectory=trajectory, ledger=ledger
        ), diagnostic_reference

    if bool(summary.get("reference_enabled", False)):
        # The exact-audit path already populated exact diagnostic curves. Preserve
        # legacy reference payloads, but add explicit side-channel semantics.
        summary["diagnostic_exact_reference_enabled"] = True
        summary["diagnostic_exact_reference_source"] = "controller_exact_audit_payload"
        diagnostic_reference = {
            "diagnostic_reference_mode": str(diagnostic_mode),
            "diagnostic_reference_enabled": True,
            "role": "diagnostic_exact_benchmark",
            "feeds_controller_decisions": False,
            "source": "controller_exact_audit_payload",
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
        }
        return _replace_result_like(
            result, summary=summary, trajectory=trajectory, ledger=ledger
        ), diagnostic_reference

    try:
        hmat_static = np.asarray(hamiltonian_matrix(controller.h_poly), dtype=complex)
        helper = RealtimeExactAuditHelper(
            h_poly=controller.h_poly,
            hmat=hmat_static,
            psi_initial=np.asarray(controller.psi_initial, dtype=complex),
            times=np.asarray(controller.times, dtype=float),
            drive_config=getattr(controller, "_drive_config", None),
            drive_profile=getattr(controller, "_drive_profile", None),
            drive_coeff_provider_exyz=getattr(controller, "_drive_coeff_provider_exyz", None),
            drive_model=getattr(controller, "_drive_model", None),
            exact_reference_cache=exact_reference_cache,
        )
        reference_payload = dict(helper.reference_payload())
        reference_payload.update(
            {
                "diagnostic_reference_mode": str(diagnostic_mode),
                "diagnostic_reference_enabled": True,
                "role": "diagnostic_exact_benchmark",
                "feeds_controller_decisions": False,
                "controller_reference_mode": summary.get("controller_reference_mode", "off"),
                "controller_reference_enabled": bool(summary.get("controller_reference_enabled", False)),
                "controller_exact_input_mode": summary.get(
                    "controller_exact_input_mode",
                    summary.get("controller_reference_mode", "off"),
                ),
                "uses_reference_for_decision": False,
                "uses_future_exact_forecast_for_decision": False,
            }
        )
        for idx, row in enumerate(trajectory):
            time_value = float(row.get("time", 0.0))
            physical_time = _finite_or_none(row.get("physical_time", None))
            if physical_time is None:
                physical_time = (
                    float(time_value)
                    if getattr(controller, "_drive_config", None) is None
                    else float(time_value) + float(getattr(controller._drive_config, "drive_t0", 0.0))
                )
            psi_exact = np.asarray(helper.state_at(float(time_value)), dtype=complex).reshape(-1)
            hmat_step = _diagnostic_exact_step_hmat(
                controller=controller,
                hmat_static=hmat_static,
                physical_time=float(physical_time),
            )
            energy_exact = float(np.real(np.vdot(psi_exact, hmat_step @ psi_exact)))
            exact_obs = controller._observable_snapshot(psi_exact)
            primary_mode = str(controller._exact_forecast_primary_density_target_mode())
            primary_exact = controller._primary_density_value_from_snapshot(exact_obs)
            site_controller = row.get("site_occupations", None)
            site_exact = [float(x) for x in np.asarray(exact_obs.get("site_occupations", ()), dtype=float).reshape(-1).tolist()]
            row.update(
                {
                    "diagnostic_exact_reference_mode": str(diagnostic_mode),
                    "diagnostic_exact_reference_enabled": True,
                    "energy_total_exact": float(energy_exact),
                    "abs_energy_total_error": _abs_error_or_none(
                        row.get("energy_total_controller", row.get("energy_total", None)),
                        energy_exact,
                    ),
                    "primary_density_exact": _finite_or_none(primary_exact),
                    "abs_primary_density_error": _abs_error_or_none(
                        row.get("primary_density", None),
                        primary_exact,
                    ),
                    "staggered_exact": _finite_or_none(exact_obs.get("staggered", None)),
                    "abs_staggered_error": _abs_error_or_none(
                        row.get("staggered", None),
                        exact_obs.get("staggered", None),
                    ),
                    "doublon_exact": _finite_or_none(exact_obs.get("doublon", None)),
                    "abs_doublon_error": _abs_error_or_none(
                        row.get("doublon", None),
                        exact_obs.get("doublon", None),
                    ),
                    "site_occupations_exact": site_exact,
                    "site_occupations_up_exact": [
                        float(x) for x in np.asarray(exact_obs.get("n_up_site", ()), dtype=float).reshape(-1).tolist()
                    ],
                    "site_occupations_dn_exact": [
                        float(x) for x in np.asarray(exact_obs.get("n_dn_site", ()), dtype=float).reshape(-1).tolist()
                    ],
                    "site_occupations_abs_error": _vector_abs_error_or_none(site_controller, site_exact),
                    "site_occupations_abs_error_max": _vector_error_max_or_none(site_controller, site_exact),
                    "primary_density_mode": str(primary_mode),
                }
            )
            if idx < len(ledger):
                ledger[idx].update(
                    {
                        "diagnostic_exact_reference_mode": str(diagnostic_mode),
                        "diagnostic_exact_reference_enabled": True,
                        "energy_total_exact": float(energy_exact),
                        "abs_energy_total_error": row.get("abs_energy_total_error"),
                    }
                )
        final_row = trajectory[-1] if trajectory else {}
        summary.update(
            {
                "diagnostic_exact_reference_enabled": True,
                "diagnostic_exact_reference_source": "postrun_exact_statevector_benchmark",
                "final_abs_energy_total_error": _finite_or_none(final_row.get("abs_energy_total_error", None)),
                "final_staggered_exact": _finite_or_none(final_row.get("staggered_exact", None)),
                "final_abs_staggered_error": _finite_or_none(final_row.get("abs_staggered_error", None)),
                "final_doublon_exact": _finite_or_none(final_row.get("doublon_exact", None)),
                "final_abs_doublon_error": _finite_or_none(final_row.get("abs_doublon_error", None)),
                "final_site_occupations_exact": final_row.get("site_occupations_exact", None),
                "final_site_occupations_abs_error_max": _finite_or_none(
                    final_row.get("site_occupations_abs_error_max", None)
                ),
            }
        )
        for key, prefix in (
            ("abs_energy_total_error", "abs_energy_total_error"),
            ("abs_primary_density_error", "abs_primary_density_error"),
            ("abs_staggered_error", "abs_staggered_error"),
            ("abs_doublon_error", "abs_doublon_error"),
            ("site_occupations_abs_error_max", "site_occupations_abs_error_max"),
        ):
            summary.update(_summary_stats_for_values(_finite_values(trajectory, key), prefix=prefix))
        return _replace_result_like(
            result, summary=summary, trajectory=trajectory, ledger=ledger
        ), reference_payload
    except Exception as exc:
        summary["diagnostic_exact_reference_enabled"] = False
        summary["diagnostic_exact_reference_error"] = f"{type(exc).__name__}: {exc}"
        diagnostic_reference = {
            "diagnostic_reference_mode": str(diagnostic_mode),
            "diagnostic_reference_enabled": False,
            "role": "diagnostic_exact_benchmark",
            "feeds_controller_decisions": False,
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "error": summary["diagnostic_exact_reference_error"],
        }
        return _replace_result_like(
            result, summary=summary, trajectory=trajectory, ledger=ledger
        ), diagnostic_reference


def build_controller_bundle_from_args(
    args: argparse.Namespace,
    *,
    exact_reference_cache: dict[str, object] | None = None,
) -> dict[str, Any]:
    artifact_json = Path(args.artifact_json).expanduser().resolve()
    spec = FixedManifoldRunSpec(
        name=str(args.run_tag),
        artifact_json=artifact_json,
        loader_mode=str(args.loader_mode),
        generator_family=str(args.generator_family),
        fallback_family=str(args.fallback_family),
        append_pool_family=str(getattr(args, "append_pool_family", "match_replay")),
    )
    loaded = load_run_context(
        spec,
        tag=str(args.run_tag),
        lock_fixed_manifold=bool(args.lock_fixed_manifold),
    )
    cfg = build_controller_config(args)
    oracle_config = build_oracle_config(args)
    n_sites = int(
        getattr(
            getattr(loaded, "cfg", None),
            "L",
            getattr(getattr(loaded.replay_context, "cfg", None), "L", 1),
        )
    )
    ordering = str(
        getattr(
            getattr(loaded, "cfg", None),
            "ordering",
            getattr(getattr(loaded.replay_context, "cfg", None), "ordering", "blocked"),
        )
    )
    drive_config = build_drive_config(args, n_sites=n_sites, ordering=ordering)
    strict_qpu_faithful = strict_qpu_faithful_requested(args)
    replay_context = loaded.replay_context
    resolved_problem = (
        None
        if getattr(loaded, "runtime_input", None) is None
        else getattr(loaded.runtime_input, "resolved_problem", None)
    )
    resolved_problem_family = _problem_family_from_loaded(
        loaded=loaded,
        replay_context=replay_context,
        explicit_problem_family=None,
    )
    if strict_qpu_faithful:
        validate_realtime_route_request(
            family_key=str(resolved_problem_family),
            controller_mode=str(cfg.mode),
            reference_mode=str(cfg.reference_mode),
            drive_requested=bool(getattr(args, "enable_drive", False)),
            strict_qpu_faithful=True,
            append_pool_family=getattr(args, "append_pool_family", "match_replay"),
            num_sites=int(n_sites),
            drive_include_identity=bool(getattr(args, "drive_include_identity", False)),
            primary_density_mode=str(
                getattr(cfg, "exact_forecast_primary_density_target_mode", "auto")
            ),
        )
    h_poly = replay_context.h_poly
    hmat = None if strict_qpu_faithful else np.asarray(hamiltonian_matrix(h_poly), dtype=complex)
    controller = RealtimeCheckpointController(
        cfg=cfg,
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=np.asarray(loaded.psi_initial, dtype=complex),
        best_theta=np.asarray(replay_context.adapt_theta_runtime, dtype=float),
        allow_repeats=bool(args.allow_repeats),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        drive_config=drive_config,
        oracle_base_config=oracle_config,
        progress_path=getattr(args, "progress_json", None),
        partial_payload_path=getattr(args, "partial_payload_json", None),
        exact_reference_cache=exact_reference_cache,
        resolved_problem=resolved_problem,
        strict_qpu_faithful=strict_qpu_faithful,
    )
    resolved_drive_config = getattr(controller, "_drive_config", drive_config)
    exact_helper = (
        None
        if strict_qpu_faithful or str(cfg.reference_mode) != "benchmark_exact"
        else build_exact_audit_helper_for_controller(
            controller,
            exact_reference_cache=exact_reference_cache,
        )
    )
    strict_qpu_hh_mirror = bool(getattr(controller, "strict_qpu_hh", strict_qpu_faithful))
    return {
        "loaded": loaded,
        "cfg": cfg,
        "drive_config": resolved_drive_config,
        "oracle_config": oracle_config,
        "controller": controller,
        "exact_helper": exact_helper,
        "strict_qpu_faithful": strict_qpu_faithful,
        "strict_qpu_hh": strict_qpu_hh_mirror,
    }


def build_output_payload(
    *,
    args: argparse.Namespace,
    loaded: Any,
    cfg: RealtimeCheckpointConfig,
    drive_config: ControllerDriveConfig | None,
    oracle_config: OracleConfig | None,
    result: Any,
    compile_audit: Mapping[str, Any] | None = None,
    ed_ground_exact_energy: float | None = None,
    ed_ground_exact_energy_source: str | None = None,
    diagnostic_reference: Mapping[str, Any] | None = None,
    problem_family: str | None = None,
) -> dict[str, Any]:
    replay_context = loaded.replay_context
    oracle_request = (
        None if oracle_config is None else normalize_oracle_execution_request(oracle_config)
    )
    oracle_capability = (
        None if oracle_config is None else assess_oracle_execution_capability(oracle_config)
    )
    summary = dict(result.summary)
    if compile_audit is not None:
        summary.update(compile_audit_summary_mirrors(compile_audit))
    resolved_problem_family = _problem_family_from_loaded(
        loaded=loaded,
        replay_context=replay_context,
        explicit_problem_family=problem_family,
    )
    strict_qpu_faithful_flag = bool(
        summary.get("strict_qpu_faithful", strict_qpu_faithful_requested(args))
    )
    strict_qpu_hh_flag = bool(
        summary.get(
            "strict_qpu_hh",
            bool(strict_qpu_faithful_flag and resolved_problem_family == "hh"),
        )
    )
    summary.setdefault("controller_exact_input_mode", str(getattr(cfg, "reference_mode", "off")))
    fallback_flow_fields = decision_data_flow_fields(
        controller_mode=str(getattr(cfg, "mode", summary.get("mode", "off"))),
        controller_exact_input_mode=str(summary.get("controller_exact_input_mode", "off")),
        decision_backend=str(summary.get("decision_backend", summary.get("requested_decision_backend", ""))),
        decision_noise_mode=(
            None
            if summary.get("decision_noise_mode", None) is None
            else str(summary.get("decision_noise_mode"))
        ),
        strict_qpu_faithful=bool(strict_qpu_faithful_flag),
        uses_reference_for_decision=bool(summary.get("uses_reference_for_decision", False)),
        uses_future_exact_forecast_for_decision=bool(
            summary.get("uses_future_exact_forecast_for_decision", False)
        ),
    )
    for key, value in fallback_flow_fields.items():
        summary.setdefault(key, value)
    payload = {
        "run_tag": str(args.run_tag),
        "artifact_json": str(Path(args.artifact_json).resolve()),
        "loader_mode": str(args.loader_mode),
        "loader_summary": {
            "generator_family": str(args.generator_family),
            "fallback_family": str(args.fallback_family),
            "append_pool_family_requested": str(
                getattr(args, "append_pool_family", "match_replay")
            ),
            "resolved_family": str(getattr(replay_context, "family_info", {}).get("resolved", "unknown")),
            "resolved_replay_family": str(
                getattr(replay_context, "family_info", {}).get("resolved", "unknown")
            ),
            "replay_family_resolution_source": str(
                getattr(replay_context, "family_info", {}).get("resolution_source", "unknown")
            ),
            "replay_family_fallback_used": bool(
                getattr(replay_context, "family_info", {}).get("fallback_used", False)
            ),
            "resolved_append_family": str(
                (getattr(replay_context, "append_family_info", None) or {}).get(
                    "resolved",
                    getattr(replay_context, "family_info", {}).get("resolved", "unknown"),
                )
            ),
            "append_family_resolution_source": str(
                (getattr(replay_context, "append_family_info", None) or {}).get(
                    "resolution_source",
                    "replay_family",
                )
            ),
            "append_family_fallback_used": bool(
                (getattr(replay_context, "append_family_info", None) or {}).get(
                    "fallback_used",
                    False,
                )
            ),
            "handoff_state_kind": str(getattr(replay_context, "handoff_state_kind", "unknown")),
            "family_terms_count": int(getattr(replay_context, "family_terms_count", 0)),
            "append_family_terms_count": (
                None
                if getattr(replay_context, "append_family_terms_count", None) is None
                else int(getattr(replay_context, "append_family_terms_count"))
            ),
            "replay_candidate_pool_complete": bool(
                getattr(replay_context, "pool_meta", {}).get("candidate_pool_complete", False)
            ),
            "append_candidate_pool_complete": bool(
                (getattr(replay_context, "append_pool_meta", None) or {}).get(
                    "candidate_pool_complete",
                    False,
                )
            ),
            "append_pool_source": (
                getattr(replay_context, "append_pool_meta", None) or {}
            ).get("append_pool_source", None),
            "adapt_depth": int(getattr(replay_context, "adapt_depth", 0)),
        },
        "hamiltonian_capabilities": _to_jsonable(
            adapter_for_family_key(resolved_problem_family).capabilities
        ),
        "route_config": {
            "route_version": ROUTE_VERSION,
            "route_authority": ROUTE_AUTHORITY,
            "route_label": ROUTE_LABEL,
            "drive_enabled": bool(drive_config is not None and drive_config.enabled),
            "enable_drive_requested": bool(getattr(args, "enable_drive", False)),
            "disable_drive_requested": bool(getattr(args, "disable_drive", False)),
            "drive_defaults_source": DRIVE_DEFAULTS_SOURCE,
            "problem_family": str(resolved_problem_family),
            "strict_qpu_faithful": strict_qpu_faithful_flag,
            "strict_qpu_hh": strict_qpu_hh_flag,
            "controller_exact_input_mode": str(getattr(cfg, "reference_mode", "off")),
            "diagnostic_exact_reference_mode": str(
                normalize_reference_mode(getattr(args, "diagnostic_exact_reference_mode", "benchmark_exact"))
            ),
            "decision_data_flow": summary.get("decision_data_flow", "unknown"),
            "uses_reference_for_decision": bool(
                summary.get("uses_reference_for_decision", False)
            ),
            "uses_future_exact_forecast_for_decision": bool(
                summary.get("uses_future_exact_forecast_for_decision", False)
            ),
            "uses_statevector_as_ideal_observable_estimator": bool(
                summary.get("uses_statevector_as_ideal_observable_estimator", False)
            ),
            "strict_measurement_oracle_certified": bool(
                summary.get("strict_measurement_oracle_certified", False)
            ),
            "compile_audit_mode": str(getattr(args, "compile_audit_mode", "off")),
            "compile_audit_local_fake_only": True,
        },
        "controller_config": _to_jsonable(cfg),
        "drive_config": _to_jsonable(drive_config),
        "oracle_config": _to_jsonable(oracle_request),
        "oracle_capability": _to_jsonable(oracle_capability),
        "logging": {
            "progress_json": (
                None
                if getattr(args, "progress_json", None) in {None, ""}
                else str(Path(str(args.progress_json)).expanduser().resolve())
            ),
            "partial_payload_json": (
                None
                if getattr(args, "partial_payload_json", None) in {None, ""}
                else str(Path(str(args.partial_payload_json)).expanduser().resolve())
            ),
        },
        "summary": _to_jsonable(summary),
        "trajectory": _to_jsonable([dict(row) for row in result.trajectory]),
        "ledger": _to_jsonable([dict(row) for row in result.ledger]),
        "reference": _to_jsonable(dict(result.reference)),
    }
    if diagnostic_reference is not None:
        payload["diagnostic_reference"] = _to_jsonable(dict(diagnostic_reference))
    if compile_audit is not None:
        payload["compile_audit"] = _to_jsonable(compile_audit)
    annotate_ed_ground_energy_target(
        payload,
        exact_energy=ed_ground_exact_energy,
        source=ed_ground_exact_energy_source,
        drive_enabled=bool(drive_config is not None and drive_config.enabled),
    )
    return payload


"Built Math: H_mat = matrix(H_poly), theta*(t) = Controller(H_mat, psi_0, theta_0), payload = {summary, trajectory, ledger}."
def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    output_json = Path(args.output_json).expanduser().resolve()
    bundle = build_controller_bundle_from_args(args)
    loaded = bundle["loaded"]
    cfg = bundle["cfg"]
    oracle_config = bundle["oracle_config"]
    resolved_drive_config = bundle["drive_config"]
    controller = bundle["controller"]
    exact_helper = bundle.get("exact_helper")
    ed_ground_exact_energy, ed_ground_exact_energy_source = _exact_energy_target_from_loaded(
        loaded
    )
    result = (
        controller.run()
        if exact_helper is None
        else run_controller_with_exact_audit(
            controller,
            exact_helper,
            ed_ground_exact_energy=ed_ground_exact_energy,
            ed_ground_exact_energy_source=ed_ground_exact_energy_source,
        )
    )
    result, diagnostic_reference = _attach_diagnostic_exact_reference(
        args=args,
        controller=controller,
        result=result,
    )
    compile_audit_config = build_compile_audit_config_from_args(args)
    compile_audit = None
    if str(compile_audit_config.mode) != "off":
        compile_audit = run_final_scaffold_compile_audit(
            controller=controller,
            config=compile_audit_config,
        )
        compile_audit = dict(compile_audit)
        compile_audit["prune_event_audit"] = run_prune_event_compile_audit(
            controller=controller,
            config=compile_audit_config,
        )
    payload = build_output_payload(
        args=args,
        loaded=loaded,
        cfg=cfg,
        drive_config=resolved_drive_config,
        oracle_config=oracle_config,
        result=result,
        compile_audit=compile_audit,
        ed_ground_exact_energy=ed_ground_exact_energy,
        ed_ground_exact_energy_source=ed_ground_exact_energy_source,
        diagnostic_reference=diagnostic_reference,
        problem_family=_problem_family_from_loaded(
            loaded=loaded,
            replay_context=loaded.replay_context,
            explicit_problem_family=None,
        ),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HH realtime checkpoint controller from an ADAPT artifact."
    )
    parser.add_argument("--artifact-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--progress-json", default=None)
    parser.add_argument("--partial-payload-json", default=None)
    parser.add_argument("--run-tag", default="hh_realtime_from_adapt_artifact")
    parser.add_argument("--loader-mode", default="replay_family")
    parser.add_argument("--generator-family", default="match_adapt")
    parser.add_argument("--fallback-family", default="full_meta")
    parser.add_argument(
        "--append-pool-family",
        "--candidate-pool-family",
        dest="append_pool_family",
        default="match_replay",
        help=(
            "Append/candidate-pool family. Default 'match_replay' preserves the "
            "artifact-resolved replay/scaffold pool; explicit values such as "
            "'full_meta' opt into a separate append surface."
        ),
    )
    parser.add_argument("--lock-fixed-manifold", action="store_true")
    parser.add_argument("--allow-repeats", action="store_true")
    parser.add_argument("--t-final", type=float, default=T_FINAL_DEFAULT)
    parser.add_argument("--num-times", type=int, default=NUM_TIMES_DEFAULT)
    drive_group = parser.add_mutually_exclusive_group()
    drive_group.add_argument(
        "--enable-drive",
        action="store_true",
        default=ENABLE_DRIVE_DEFAULT,
        help="Enable the opt-in time-dependent onsite density drive.",
    )
    drive_group.add_argument(
        "--disable-drive",
        action="store_true",
        help="Deprecated compatibility flag; no flag is already static/no-drive.",
    )
    parser.add_argument("--drive-A", type=float, default=DRIVE_A_DEFAULT)
    parser.add_argument("--drive-omega", type=float, default=DRIVE_OMEGA_DEFAULT)
    parser.add_argument("--drive-tbar", type=float, default=DRIVE_TBAR_DEFAULT)
    parser.add_argument("--drive-phi", type=float, default=DRIVE_PHI_DEFAULT)
    parser.add_argument("--drive-pattern", default=DRIVE_PATTERN_DEFAULT)
    parser.add_argument("--drive-custom-weights", default="")
    parser.add_argument("--drive-include-identity", action="store_true")
    parser.add_argument("--drive-time-sampling", default=DRIVE_TIME_SAMPLING_DEFAULT)
    parser.add_argument("--drive-t0", type=float, default=DRIVE_T0_DEFAULT)
    parser.add_argument("--exact-steps-multiplier", type=int, default=EXACT_STEPS_MULTIPLIER_DEFAULT)
    parser.add_argument("--checkpoint-controller-mode", default="exact_v1")
    parser.add_argument(
        "--checkpoint-controller-strict-qpu-faithful",
        action="store_true",
        help=(
            "Strict QPU-faithful route: controller exact inputs off, "
            "measurement-compatible observable decisions, no exact fallback. "
            "Use observable_v1 for the fast ideal-observable statevector estimator "
            "or oracle_v1 for the measured oracle surface. Exact diagnostic overlays "
            "are controlled separately by --diagnostic-exact-reference-mode."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-strict-qpu-hh",
        action="store_true",
        help=(
            "Legacy alias for --checkpoint-controller-strict-qpu-faithful retained "
            "for existing HH strict-route scripts."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-reference-mode",
        choices=("off", "benchmark_exact", "benchmark", "exact", "disabled"),
        default="off",
        help=(
            "Controller exact-input mode. 'off' means exact target/reference data "
            "is unavailable to decision logic; it does not disable diagnostic exact plots."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-exact-input-mode",
        dest="checkpoint_controller_reference_mode",
        choices=("off", "benchmark_exact", "benchmark", "exact", "disabled"),
        help=(
            "Clear alias for --checkpoint-controller-reference-mode. Controls exact "
            "target/reference inputs to controller decisions, not report overlays."
        ),
    )
    parser.add_argument(
        "--diagnostic-exact-reference-mode",
        choices=("off", "benchmark_exact", "benchmark", "exact", "disabled"),
        default="benchmark_exact",
        help=(
            "Post-run/report exact benchmark mode. Defaults to benchmark_exact so "
            "strict QPU-faithful runs include exact overlays without exposing them to decisions."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-oracle-selection-policy",
        choices=["measured_gain_commit_veto", "measured_topk_oracle_energy"],
        default="measured_gain_commit_veto",
    )
    parser.add_argument(
        "--checkpoint-controller-noise-mode",
        choices=["ideal", "shots", "aer_noise", "aer_density_matrix", "backend_scheduled", "runtime"],
        default=None,
    )
    parser.add_argument(
        "--checkpoint-controller-value-noise-model",
        choices=["off", "gaussian_iid_v1"],
        default="off",
        help="Opt-in post-expectation value noise for checkpoint-controller oracle_v1 decisions; not physical shots.",
    )
    parser.add_argument("--checkpoint-controller-value-noise-std", type=float, default=0.0)
    parser.add_argument("--checkpoint-controller-value-noise-seed", type=int, default=None)
    parser.add_argument("--backend-name", default=None)
    parser.add_argument("--use-fake-backend", action="store_true")
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--oracle-repeats", type=int, default=1)
    parser.add_argument("--oracle-aggregate", choices=["mean"], default="mean")
    parser.add_argument(
        "--allow-aer-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seed-transpiler", type=int, default=0)
    parser.add_argument("--transpile-optimization-level", type=int, default=2)
    parser.add_argument(
        "--compile-audit-mode",
        choices=("off", "final_scaffold"),
        default="off",
        help="Opt-in local fake-backend compile audit for the final realtime scaffold.",
    )
    parser.add_argument("--compile-audit-backend-name", default="FakeMarrakesh")
    parser.add_argument("--compile-audit-seed-transpiler", type=int, default=7)
    parser.add_argument("--compile-audit-optimization-level", type=int, default=2)
    parser.add_argument("--compile-audit-export-circuit-dir", default=None)
    parser.add_argument(
        "--compile-audit-preferred-fake-backends",
        default="FakeMarrakesh,FakeNighthawk,FakeFez",
        help="Comma-separated local fake fallback shortlist. Runtime lookup is disabled for this audit.",
    )
    parser.add_argument("--checkpoint-controller-runtime-profile", default=None)
    parser.add_argument("--checkpoint-controller-runtime-raw-profile", default=None)
    parser.add_argument("--checkpoint-controller-runtime-session-policy", default=None)
    parser.add_argument("--checkpoint-controller-raw-transport", default="auto")
    parser.add_argument("--checkpoint-controller-raw-store-memory", action="store_true")
    parser.add_argument("--checkpoint-controller-raw-artifact-path", default=None)
    parser.add_argument("--final-noise-audit-local-readout-strategy", default=None)
    parser.add_argument(
        "--final-noise-audit-local-gate-twirling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--final-noise-audit-dd-sequence", default=None)
    parser.add_argument("--final-noise-audit-zne-scales", default=None)
    parser.add_argument("--final-noise-audit-zne-extrapolator", default=None)
    parser.add_argument("--final-noise-audit-runtime-profile", default=None)
    parser.add_argument("--final-noise-audit-runtime-session-policy", default=None)
    parser.add_argument("--checkpoint-controller-miss-threshold", type=float, default=0.05)
    parser.add_argument(
        "--checkpoint-controller-high-miss-no-admit-policy",
        choices=("bounded_stay_advance", "legacy_advance_stay", "repair_stop", "repair_retry"),
        default=HIGH_MISS_NO_ADMIT_POLICY_DEFAULT,
    )
    parser.add_argument("--checkpoint-controller-repair-retry-max-attempts", type=int, default=2)
    parser.add_argument(
        "--checkpoint-controller-repair-retry-escalation-mode",
        choices=("append_budget_then_stabilize_v1",),
        default="append_budget_then_stabilize_v1",
    )
    parser.add_argument(
        "--checkpoint-controller-repair-retry-admission-policy",
        choices=("strict", "rescue_best_confirmed_append_v1"),
        default="strict",
    )
    parser.add_argument(
        "--checkpoint-controller-repair-retry-rescue-min-gain-ratio",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-repair-retry-rescue-attempt",
        choices=("terminal_attempt_only",),
        default="terminal_attempt_only",
    )
    parser.add_argument("--checkpoint-controller-miss-abs-threshold", type=float, default=0.0)
    parser.add_argument("--checkpoint-controller-miss-persistence-window", type=int, default=1)
    parser.add_argument("--checkpoint-controller-miss-persistence-count", type=int, default=1)
    parser.add_argument(
        "--checkpoint-controller-integrator-policy",
        choices=("euler", "rk4", "auto_euler_rk4"),
        default="auto_euler_rk4",
    )
    parser.add_argument(
        "--checkpoint-controller-integrator-columnarity-threshold",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--checkpoint-controller-integrator-curvature-threshold",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--checkpoint-controller-integrator-euler-fs-error-threshold",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--checkpoint-controller-integrator-condition-max",
        type=float,
        default=1.0e12,
    )
    parser.add_argument(
        "--checkpoint-controller-integrator-euler-min-time-fraction",
        type=float,
        default=0.35,
        help=(
            "For auto_euler_rk4, suppress Euler until this fraction of the "
            "trajectory has elapsed. This implements the early-RK4/late-Euler "
            "time prior; Euler still also requires the Chapter 17A calm/columnar gates."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-integrator-euler-observable-window",
        type=int,
        default=16,
        help=(
            "For auto_euler_rk4, number of recent physical controller rows used "
            "to decide whether observables are calm enough for Euler."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-integrator-euler-site-span-max",
        type=float,
        default=None,
        help=(
            "Optional maximum recent site-occupation span allowed before auto Euler may be used. "
            "Omit to disable this additional observable-calm gate."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-integrator-euler-primary-density-span-max",
        type=float,
        default=None,
        help=(
            "Optional maximum recent primary-density span allowed before auto Euler may be used. "
            "Omit to disable this additional observable-calm gate."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-integrator-euler-energy-span-max",
        type=float,
        default=None,
        help="Optional recent controller-energy span cap before auto Euler may be used.",
    )
    parser.add_argument("--checkpoint-controller-gain-ratio-threshold", type=float, default=0.02)
    parser.add_argument("--checkpoint-controller-append-margin-abs", type=float, default=1e-6)
    parser.add_argument(
        "--checkpoint-controller-append-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable checkpoint append candidate generation. Use --no-checkpoint-controller-append-enabled "
            "only for explicit no-append ablation runs; exact target/reference data remains governed "
            "separately by --checkpoint-controller-exact-input-mode."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-append-no-harm-guard-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the measurement-compatible append no-harm veto. This guard may inspect "
            "prepared-state observable/geometry diagnostics, not exact target/reference trajectories."
        ),
    )
    parser.add_argument("--checkpoint-controller-append-no-harm-condition-ratio-cap", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-append-no-harm-displacement-ratio-cap", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-append-no-harm-condition-abs-floor", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-append-no-harm-kink-min-step-gain-delta", type=float, default=1.0e-3, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-append-no-harm-kink-max-condition-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-append-no-harm-kink-max-displacement-ratio", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-append-no-harm-rho-only-min-step-gain-delta", type=float, default=1.0e-3, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-append-no-harm-rho-only-condition-ratio-cap", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-append-no-harm-rho-only-step-residual-ratio-cap", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-append-no-harm-rho-only-displacement-ratio-cap", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-shortlist-size", type=int, default=4)
    parser.add_argument("--checkpoint-controller-shortlist-fraction", type=float, default=0.15)
    parser.add_argument("--checkpoint-controller-active-window-size", type=int, default=3)
    parser.add_argument(
        "--checkpoint-controller-measurement-active-window-size",
        type=int,
        default=0,
        help=(
            "Opt-in strict ideal oracle measured-geometry active window in logical blocks. "
            "0 disables compact measurement planning."
        ),
    )
    parser.add_argument("--checkpoint-controller-max-probe-positions", type=int, default=4)
    parser.add_argument("--checkpoint-controller-regularization-lambda", type=float, default=1e-8)
    parser.add_argument("--checkpoint-controller-candidate-regularization-lambda", type=float, default=1e-8)
    parser.add_argument("--checkpoint-controller-pinv-rcond", type=float, default=1e-10)
    parser.add_argument("--checkpoint-controller-compile-penalty-weight", type=float, default=0.05)
    parser.add_argument("--checkpoint-controller-measurement-penalty-weight", type=float, default=0.02)
    parser.add_argument("--checkpoint-controller-directional-penalty-weight", type=float, default=0.01)
    parser.add_argument(
        "--checkpoint-controller-confirm-score-mode",
        choices=("exact_gain_ratio", "compressed_whitened_v1"),
        default="compressed_whitened_v1",
    )
    parser.add_argument("--checkpoint-controller-confirm-compress-fraction", type=float, default=0.5)
    parser.add_argument("--checkpoint-controller-confirm-compress-min-modes", type=int, default=1)
    parser.add_argument("--checkpoint-controller-confirm-compress-max-modes", type=int, default=8)
    parser.add_argument("--checkpoint-controller-prune-mode", default="off")
    parser.add_argument("--checkpoint-controller-prune-miss-threshold", type=float, default=0.02)
    parser.add_argument("--checkpoint-controller-prune-loss-threshold", type=float, default=0.01)
    parser.add_argument("--checkpoint-controller-prune-theta-block-tol", type=float, default=0.05)
    parser.add_argument("--checkpoint-controller-prune-state-jump-l2-tol", type=float, default=0.05)
    parser.add_argument(
        "--checkpoint-controller-prune-safe-miss-increase-tol",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--checkpoint-controller-prune-no-harm-guard-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require a same-integrator local-projective no-harm verification before "
            "accepting a prune. This uses measurement-compatible controller observables/geometry, "
            "not exact target-state data."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-prune-no-harm-score-increase-tol",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-controller-prune-no-harm-step-residual-ratio-increase-tol",
        type=float,
        default=1.0e-6,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--checkpoint-controller-prune-schur-ladder-local-radius", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-prune-schur-monotonicity-tol", type=float, default=1.0e-9, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-prune-loss-norm-epsilon", type=float, default=1.0e-14, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-prune-differential-miss-tol", type=float, default=1.0e-2, help=argparse.SUPPRESS)
    parser.add_argument(
        "--checkpoint-controller-prune-high-miss-differential-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-controller-prune-projection-mode",
        choices=("state_tangent_ls_v1", "raw_delete"),
        default="state_tangent_ls_v1",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--checkpoint-controller-prune-projection-rounds", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-prune-projection-max-active-runtime", type=int, default=64, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-prune-projection-trust-radius", type=float, default=5.0e-2, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-prune-projection-regularization", type=float, default=1.0e-8, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-prune-ray-distance-tol", type=float, default=5.0e-2, help=argparse.SUPPRESS)
    parser.add_argument(
        "--checkpoint-controller-prune-shadow-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--checkpoint-controller-prune-shadow-horizon-steps", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-prune-shadow-score-increase-tol", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-prune-persistence-window", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-controller-prune-persistence-required", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--checkpoint-controller-prune-appended-origin-bias-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-controller-prune-appended-origin-target-policy",
        choices=("append_only", "prefer_append", "bias_only"),
        default="append_only",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-controller-prune-appended-origin-grace-steps",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-controller-prune-initial-scaffold-grace-steps",
        type=int,
        default=64,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-controller-prune-appended-origin-bias-scale",
        type=float,
        default=0.10,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-controller-prune-appended-origin-bias-max-factor",
        type=float,
        default=0.50,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-controller-candidate-step-scales",
        default="0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.8,1.0",
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-baseline-step-refine-rounds",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-baseline-proposal-mode",
        choices=("norm_locked_blend_v1", "anticipatory_drive_basis_v1"),
        default="norm_locked_blend_v1",
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-baseline-blend-weights",
        default="-0.25,-0.125,0.0,0.125,0.25,0.375,0.5,0.75,1.0",
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-baseline-gain-scales",
        default="",
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-include-tangent-secant-proposal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tangent-secant-trust-radius",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tangent-secant-signed-energy-lead-limit",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-horizon-steps",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-horizon-weights",
        default="2.0,1.0",
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-primary-density-target-mode",
        choices=("auto", "pair_difference", "staggered"),
        default="auto",
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tracking-fidelity-defect-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tracking-primary-density-error-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tracking-staggered-error-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tracking-doublon-error-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tracking-site-occupations-error-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-tracking-energy-total-error-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-density-slope-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-density-curvature-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-density-excursion-under-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-density-excursion-over-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-density-sign-lag-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-density-postcross-wrong-sign-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-drive-harmonic-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-energy-slope-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-energy-curvature-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-energy-excursion-under-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-energy-excursion-over-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-energy-excursion-rel-tolerance",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-repeat-reopen-mode",
        choices=("off", "sign_reversal_window"),
        default="off",
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-density-first-target-gain-floor",
        type=float,
        default=2.0e-2,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-below-floor-probe-target-gain-floor",
        type=float,
        default=3.0e-2,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-sign-lag-window-activation",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-sign-lag-window-target-gain-floor",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-postcross-wrong-sign-activation",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-postcross-wrong-sign-target-gain-floor",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-postcross-compare-diag",
        action="store_true",
        help="Persist exact_v1 stay/selected/runner-up postcross score diagnostics.",
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-below-floor-energy-safe-turn-escape",
        action="store_true",
        help=(
            "Allow a below-floor probe to bypass the blanket outside-energy-safe-window "
            "rejection when it improves total tracking score, combined per-site turn telemetry, "
            "and does not worsen raw next-step energy."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-below-floor-energy-safe-d-shape-escape",
        action="store_true",
        help=(
            "Allow a below-floor probe to bypass the blanket outside-energy-safe-window "
            "rejection when it improves total tracking score, the shadow d-shape total "
            "(curvature plus excursion under/over), and does not worsen raw next-step energy "
            "or total occupation N."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-d-shape-turn-window-abs-activation",
        type=float,
        default=0.0,
        help=(
            "Optional exact-only min-|d| horizon activation for the d-shape protected-horizon "
            "bridge. The bridge is active on exact zero-cross horizons, or when the exact "
            "horizon min |d| falls below this threshold while still moving closer to zero."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-d-shape-outside-turn-below-floor-probe-stall-threshold",
        type=int,
        default=0,
        help=(
            "Optional default-off stall-streak override for below-floor probing when "
            "d_shape_barrier_v1 is active but the exact turn window is not. Use this to "
            "delay old componentwise aspiration retries before the true turn window."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-d-shape-pre-turn-shadow-bridge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Default-off protected-horizon bridge for d_shape_barrier_v1 before the exact turn "
            "window is active. It only opens when the exact horizon is still moving toward the turn, "
            "the candidate beats stay on total tracking score and shadow d-shape total, and the usual "
            "energy/fidelity/total-occupation guardrails still pass."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-exact-v1-single-surface-commit-law",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimental A/B seam: use the same full forecast tracking surface for guarded "
            "componentwise aspiration admission and final commit-vs-stay override decisions."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-guardrail-mode",
        choices=(
            "off",
            "dual_metric_v1",
            "d_shape_barrier_v1",
            "fidelity_first_barrier_v1",
        ),
        default="off",
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-fidelity-loss-tol",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-abs-energy-error-increase-tol",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-exact-forecast-total-occupation-error-increase-tol",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-progress-observable-window",
        type=int,
        default=16,
        help="Rolling window size for progress heartbeat exact-delta means.",
    )
    parser.add_argument(
        "--checkpoint-controller-progress-early-stop-min-checkpoint",
        type=int,
        default=0,
        help="Minimum completed checkpoint index before opt-in progress early-stop thresholds can trigger.",
    )
    parser.add_argument(
        "--checkpoint-controller-progress-early-stop-site-error-mean-max",
        type=float,
        default=None,
        help="Optional rolling-mean max-site-occupation exact-error threshold for early stop.",
    )
    parser.add_argument(
        "--checkpoint-controller-progress-early-stop-primary-density-error-mean-max",
        type=float,
        default=None,
        help="Optional rolling-mean primary-density exact-error threshold for early stop.",
    )
    parser.add_argument(
        "--checkpoint-controller-progress-early-stop-energy-error-mean-max",
        type=float,
        default=None,
        help="Optional rolling-mean total-energy exact-error threshold for early stop.",
    )
    parser.add_argument(
        "--checkpoint-controller-progress-early-stop-site-span-max",
        type=float,
        default=None,
        help=(
            "Optional max rolling span of measured site occupations for observable-stability "
            "early stop; uses controller observables, not exact-reference errors."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-progress-early-stop-primary-density-span-max",
        type=float,
        default=None,
        help=(
            "Optional rolling span of measured primary density/staggered observable for "
            "observable-stability early stop."
        ),
    )
    parser.add_argument(
        "--checkpoint-controller-progress-early-stop-energy-span-max",
        type=float,
        default=None,
        help=(
            "Optional rolling span of measured total energy for observable-stability early stop."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_from_args(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
