"""Pure oracle/noise helpers for static ADAPT.

This module intentionally does not import runtime oracle, Qiskit, or pipeline
execution machinery. It owns serialization, validation, exact-structure guards,
uncertainty helpers, and noise-floor telemetry payloads only.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from pipelines.exact_bench.noise_oracle_defaults import (
    SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT,
)
from pipelines.static_adapt.cli_config import (
    FinalNoiseAuditConfig,
    Phase3OracleGradientConfig,
    _oracle_mitigation_payload_from_fields,
    _resolve_final_noise_audit_config,
    _resolve_phase3_oracle_gradient_config,
    _value_noise_payload_from_fields,
)

__all__ = [
    "_json_ready",
    "_validate_oracle_execution_request_via_bindings",
    "_phase3_oracle_mitigation_payload",
    "_final_noise_audit_config_payload",
    "_phase3_oracle_gradient_config_payload",
    "_phase3_oracle_inner_zero_noise_exact_equivalent",
    "_phase3_oracle_inner_value_noise_exact_structure_eligible",
    "_estimate_stderr_value",
    "_oracle_fd_gradient_stderr",
    "_phase3_sigma_hat_for_label",
    "_finite_float_or_none",
    "_noise_floor_snapshot_dict",
    "_noise_floor_gradient_snr_values",
    "_noise_floor_agreement_v1_payload",
]


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _validate_oracle_execution_request_via_bindings(
    bindings: Mapping[str, Any],
    oracle_config: Any,
) -> dict[str, Any] | None:
    validate_fn = bindings.get("validate_oracle_execution_request")
    if callable(validate_fn):
        return _json_ready(validate_fn(oracle_config))
    fallback_validate_fn = bindings.get("validate_controller_oracle_base_config")
    if callable(fallback_validate_fn):
        fallback_validate_fn(oracle_config)
    normalize_fn = bindings.get("normalize_oracle_execution_request")
    if callable(normalize_fn):
        return {
            "supported": True,
            "reason_code": "ok",
            "reason": "ok",
            "normalized_request": _json_ready(normalize_fn(oracle_config)),
        }
    return None


def _phase3_oracle_mitigation_payload(config: Phase3OracleGradientConfig) -> dict[str, Any]:
    return _oracle_mitigation_payload_from_fields(
        mitigation_mode=str(config.mitigation_mode),
        local_readout_strategy=config.local_readout_strategy,
        zne_scales=tuple(getattr(config, "zne_scales", ()) or ()),
        dd_sequence=getattr(config, "dd_sequence", None),
        local_gate_twirling=bool(getattr(config, "local_gate_twirling", False)),
    )


def _final_noise_audit_config_payload(
    config: FinalNoiseAuditConfig | None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    config = _resolve_final_noise_audit_config(config)
    return {
        "noise_mode": str(config.noise_mode),
        "shots": int(config.shots),
        "oracle_repeats": int(config.oracle_repeats),
        "oracle_aggregate": str(config.oracle_aggregate),
        "backend_name": (None if config.backend_name in {None, ""} else str(config.backend_name)),
        "use_fake_backend": bool(config.use_fake_backend),
        "seed": int(config.seed),
        "mitigation": dict(
            _oracle_mitigation_payload_from_fields(
                mitigation_mode=str(config.mitigation_mode),
                local_readout_strategy=config.local_readout_strategy,
                zne_scales=tuple(getattr(config, "zne_scales", ()) or ()),
                dd_sequence=getattr(config, "dd_sequence", None),
                local_gate_twirling=bool(getattr(config, "local_gate_twirling", False)),
            )
        ),
        "runtime_profile": {"name": str(config.runtime_profile_name)},
        "runtime_session": {"mode": str(config.runtime_session_policy)},
        "compare_unmitigated_baseline": bool(config.compare_unmitigated_baseline),
        "execution_surface": "expectation_v1",
        "seed_transpiler": config.seed_transpiler,
        "transpile_optimization_level": int(config.transpile_optimization_level),
        "strict": bool(config.strict),
        "value_noise": _value_noise_payload_from_fields(
            value_noise_model=str(getattr(config, "value_noise_model", "off")),
            value_noise_std=float(getattr(config, "value_noise_std", 0.0)),
            value_noise_seed=getattr(config, "value_noise_seed", None),
            value_noise_sigma0_abs=getattr(config, "value_noise_sigma0_abs", None),
            value_noise_n_eff=getattr(config, "value_noise_n_eff", None),
            value_noise_semantic=getattr(config, "value_noise_semantic", None),
            value_noise_std_source=getattr(config, "value_noise_std_source", None),
        ),
        "synthetic_depolarizing": {
            "one_qubit_error": float(getattr(config, "synthetic_depolarizing_1q_error", 0.0)),
            "two_qubit_error": float(getattr(config, "synthetic_depolarizing_2q_error", 0.0)),
            "one_qubit_gates": [
                str(g) for g in tuple(getattr(config, "synthetic_depolarizing_1q_gates", ()) or ())
            ],
            "two_qubit_gates": [
                str(g) for g in tuple(getattr(config, "synthetic_depolarizing_2q_gates", ()) or ())
            ],
        },
        "synthetic_coherent": {
            "one_qubit_angle_std": float(getattr(config, "synthetic_coherent_1q_angle_std", 0.0)),
            "two_qubit_angle_std": float(getattr(config, "synthetic_coherent_2q_angle_std", 0.0)),
            "seed": getattr(config, "synthetic_coherent_seed", None),
            "generator_mode": str(
                getattr(config, "synthetic_coherent_generator_mode", SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT)
            ),
            "one_qubit_gates": [
                str(g) for g in tuple(getattr(config, "synthetic_coherent_1q_gates", ()) or ())
            ],
            "two_qubit_gates": [
                str(g) for g in tuple(getattr(config, "synthetic_coherent_2q_gates", ()) or ())
            ],
        },
    }


def _phase3_oracle_gradient_config_payload(
    config: Phase3OracleGradientConfig | None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    config = _resolve_phase3_oracle_gradient_config(config)
    return {
        "noise_mode": str(config.noise_mode),
        "shots": int(config.shots),
        "oracle_repeats": int(config.oracle_repeats),
        "oracle_aggregate": str(config.oracle_aggregate),
        "backend_name": (None if config.backend_name in {None, ""} else str(config.backend_name)),
        "use_fake_backend": bool(config.use_fake_backend),
        "seed": int(config.seed),
        "gradient_step": float(config.gradient_step),
        "mitigation": dict(_phase3_oracle_mitigation_payload(config)),
        "scope": str(config.scope),
        "execution_surface_requested": str(config.execution_surface_requested),
        "execution_surface": str(config.execution_surface),
        "raw_transport": str(config.raw_transport),
        "raw_store_memory": bool(config.raw_store_memory),
        "raw_artifact_path": config.raw_artifact_path,
        "seed_transpiler": config.seed_transpiler,
        "transpile_optimization_level": int(config.transpile_optimization_level),
        "value_noise": _value_noise_payload_from_fields(
            value_noise_model=str(getattr(config, "value_noise_model", "off")),
            value_noise_std=float(getattr(config, "value_noise_std", 0.0)),
            value_noise_seed=getattr(config, "value_noise_seed", None),
            value_noise_sigma0_abs=getattr(config, "value_noise_sigma0_abs", None),
            value_noise_n_eff=getattr(config, "value_noise_n_eff", None),
            value_noise_semantic=getattr(config, "value_noise_semantic", None),
            value_noise_std_source=getattr(config, "value_noise_std_source", None),
        ),
        "synthetic_depolarizing": {
            "one_qubit_error": float(getattr(config, "synthetic_depolarizing_1q_error", 0.0)),
            "two_qubit_error": float(getattr(config, "synthetic_depolarizing_2q_error", 0.0)),
            "one_qubit_gates": [
                str(g) for g in tuple(getattr(config, "synthetic_depolarizing_1q_gates", ()) or ())
            ],
            "two_qubit_gates": [
                str(g) for g in tuple(getattr(config, "synthetic_depolarizing_2q_gates", ()) or ())
            ],
        },
        "synthetic_coherent": {
            "one_qubit_angle_std": float(getattr(config, "synthetic_coherent_1q_angle_std", 0.0)),
            "two_qubit_angle_std": float(getattr(config, "synthetic_coherent_2q_angle_std", 0.0)),
            "seed": getattr(config, "synthetic_coherent_seed", None),
            "generator_mode": str(
                getattr(config, "synthetic_coherent_generator_mode", SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT)
            ),
            "one_qubit_gates": [
                str(g) for g in tuple(getattr(config, "synthetic_coherent_1q_gates", ()) or ())
            ],
            "two_qubit_gates": [
                str(g) for g in tuple(getattr(config, "synthetic_coherent_2q_gates", ()) or ())
            ],
        },
    }


def _phase3_oracle_inner_zero_noise_exact_equivalent(
    config: Phase3OracleGradientConfig | None,
) -> bool:
    """Return True when requested noisy_v1 inner energies are exactly noiseless.

    This is intentionally conservative: only expectation-surface, post-value-noise-off,
    synthetic-error-zero oracle configurations may collapse back to the exact selected-energy
    path. Shot, runtime, backend-scheduled, raw-measurement, and positive synthetic/value
    noise surfaces must keep the noisy_v1 path.
    """
    if config is None:
        return False
    required_attrs = (
        "execution_surface",
        "value_noise_model",
        "value_noise_std",
        "value_noise_sigma0_abs",
        "value_noise_n_eff",
        "synthetic_depolarizing_1q_error",
        "synthetic_depolarizing_2q_error",
        "synthetic_coherent_1q_angle_std",
        "synthetic_coherent_2q_angle_std",
        "noise_mode",
        "mitigation_mode",
        "zne_scales",
        "local_gate_twirling",
        "dd_sequence",
    )
    if any(not hasattr(config, attr) for attr in required_attrs):
        return False
    try:
        execution_surface = str(getattr(config, "execution_surface")).strip().lower()
        if execution_surface != "expectation_v1":
            return False
        value_noise_model = str(getattr(config, "value_noise_model", "off")).strip().lower() or "off"
        value_noise_std = float(getattr(config, "value_noise_std", 0.0))
        if value_noise_model != "off" or value_noise_std != 0.0:
            return False
        if getattr(config, "value_noise_sigma0_abs", None) not in {None, ""}:
            return False
        if getattr(config, "value_noise_n_eff", None) not in {None, ""}:
            return False
        synthetic_1q = float(getattr(config, "synthetic_depolarizing_1q_error", 0.0))
        synthetic_2q = float(getattr(config, "synthetic_depolarizing_2q_error", 0.0))
        if synthetic_1q != 0.0 or synthetic_2q != 0.0:
            return False
        coherent_1q = float(getattr(config, "synthetic_coherent_1q_angle_std", 0.0))
        coherent_2q = float(getattr(config, "synthetic_coherent_2q_angle_std", 0.0))
        if coherent_1q != 0.0 or coherent_2q != 0.0:
            return False
        noise_mode = str(getattr(config, "noise_mode", "ideal")).strip().lower() or "ideal"
        mitigation_mode = str(getattr(config, "mitigation_mode", "none")).strip().lower() or "none"
        if mitigation_mode != "none":
            return False
        if tuple(getattr(config, "zne_scales", ()) or ()):
            return False
        if bool(getattr(config, "local_gate_twirling", False)):
            return False
        dd_sequence = getattr(config, "dd_sequence", None)
        if dd_sequence not in {None, "", "none"}:
            return False
    except (TypeError, ValueError):
        return False
    return noise_mode in {
        "ideal",
        "aer_density_matrix_synthetic_depolarizing",
        "aer_density_matrix_synthetic_coherent",
    }


def _phase3_oracle_inner_value_noise_exact_structure_eligible(
    config: Phase3OracleGradientConfig | None,
) -> bool:
    """Return True when noisy_v1 can preserve exact structure and add scalar value noise only."""

    if config is None:
        return False
    required_attrs = (
        "execution_surface",
        "value_noise_model",
        "value_noise_std",
        "synthetic_depolarizing_1q_error",
        "synthetic_depolarizing_2q_error",
        "synthetic_coherent_1q_angle_std",
        "synthetic_coherent_2q_angle_std",
        "noise_mode",
        "mitigation_mode",
        "zne_scales",
        "local_gate_twirling",
        "dd_sequence",
    )
    if any(not hasattr(config, attr) for attr in required_attrs):
        return False
    try:
        execution_surface = str(getattr(config, "execution_surface")).strip().lower()
        if execution_surface != "expectation_v1":
            return False
        value_noise_model = str(getattr(config, "value_noise_model", "off")).strip().lower() or "off"
        value_noise_std = float(getattr(config, "value_noise_std", 0.0))
        if value_noise_model != "gaussian_iid_v1" or not math.isfinite(value_noise_std) or value_noise_std <= 0.0:
            return False
        synthetic_1q = float(getattr(config, "synthetic_depolarizing_1q_error", 0.0))
        synthetic_2q = float(getattr(config, "synthetic_depolarizing_2q_error", 0.0))
        if synthetic_1q != 0.0 or synthetic_2q != 0.0:
            return False
        coherent_1q = float(getattr(config, "synthetic_coherent_1q_angle_std", 0.0))
        coherent_2q = float(getattr(config, "synthetic_coherent_2q_angle_std", 0.0))
        if coherent_1q != 0.0 or coherent_2q != 0.0:
            return False
        noise_mode = str(getattr(config, "noise_mode", "ideal")).strip().lower() or "ideal"
        mitigation_mode = str(getattr(config, "mitigation_mode", "none")).strip().lower() or "none"
        if mitigation_mode != "none":
            return False
        if tuple(getattr(config, "zne_scales", ()) or ()):
            return False
        if bool(getattr(config, "local_gate_twirling", False)):
            return False
        dd_sequence = getattr(config, "dd_sequence", None)
        if dd_sequence not in {None, "", "none"}:
            return False
    except (TypeError, ValueError):
        return False
    return noise_mode in {
        "ideal",
        "aer_density_matrix_synthetic_depolarizing",
        "aer_density_matrix_synthetic_coherent",
    }


def _estimate_stderr_value(estimate: Any) -> float:
    if isinstance(estimate, Mapping):
        raw_value = estimate.get("stderr")
    else:
        raw_value = getattr(estimate, "stderr", None)
    if raw_value is None:
        raise ValueError("Oracle estimate must expose a finite nonnegative stderr.")
    stderr_value = float(raw_value)
    if (not math.isfinite(stderr_value)) or stderr_value < 0.0:
        raise ValueError("Oracle estimate stderr must be finite and nonnegative.")
    return float(stderr_value)


def _oracle_fd_gradient_stderr(
    e_plus: Any,
    e_minus: Any,
    *,
    grad_step: float,
) -> float:
    step = float(grad_step)
    if (not math.isfinite(step)) or step <= 0.0:
        raise ValueError("grad_step must be finite and > 0 for oracle finite-difference stderr.")
    stderr_plus = _estimate_stderr_value(e_plus)
    stderr_minus = _estimate_stderr_value(e_minus)
    grad_stderr = math.sqrt(stderr_plus ** 2 + stderr_minus ** 2) / (2.0 * step)
    if (not math.isfinite(grad_stderr)) or grad_stderr < 0.0:
        raise ValueError("Resolved oracle finite-difference stderr must be finite and nonnegative.")
    return float(grad_stderr)


def _phase3_sigma_hat_for_label(
    *,
    candidate_label: str,
    sigma_by_label: Mapping[str, float],
    phase3_enabled: bool,
) -> float:
    if not bool(phase3_enabled):
        return 0.0
    sigma_raw = sigma_by_label.get(str(candidate_label))
    if sigma_raw is None:
        return 0.0
    sigma_value = float(sigma_raw)
    if (not math.isfinite(sigma_value)) or sigma_value < 0.0:
        return 0.0
    return float(sigma_value)


def _finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return float(out) if math.isfinite(out) else None


def _noise_floor_snapshot_dict(snapshot: Any | None) -> dict[str, Any] | None:
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    if snapshot is None:
        return None
    raw_dict = getattr(snapshot, "__dict__", None)
    if isinstance(raw_dict, Mapping):
        return dict(raw_dict)
    return None


def _noise_floor_gradient_snr_values(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | str | int | None]]:
    out: list[dict[str, float | str | int | None]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        grad_abs = _finite_float_or_none(row.get("gradient_abs"))
        if grad_abs is None:
            grad_abs = _finite_float_or_none(row.get("grad_abs"))
        if grad_abs is None:
            grad_signed = _finite_float_or_none(row.get("gradient"))
            if grad_signed is None:
                grad_signed = _finite_float_or_none(row.get("gradient_signed"))
            if grad_signed is not None:
                grad_abs = abs(float(grad_signed))
        sigma = _finite_float_or_none(row.get("sigma_hat"))
        sigma_source = "sigma_hat"
        if sigma is None or sigma <= 0.0:
            sigma = _finite_float_or_none(row.get("gradient_stderr"))
            sigma_source = "gradient_stderr"
        if sigma is None or sigma <= 0.0 or grad_abs is None:
            continue
        out.append(
            {
                "candidate_pool_index": (
                    int(row.get("candidate_pool_index"))
                    if row.get("candidate_pool_index") is not None
                    else None
                ),
                "gradient_abs": float(abs(float(grad_abs))),
                "sigma": float(sigma),
                "sigma_source": str(sigma_source),
                "snr": float(abs(float(grad_abs)) / float(sigma)),
            }
        )
    return out


def _noise_floor_agreement_v1_payload(
    *,
    policy: str,
    drop_plateau_gate: bool,
    drop_policy_enabled: bool,
    depth_local: int,
    drop_plateau_hits: int,
    adapt_drop_patience: int,
    controller_snapshot: Any | None,
    candidate_gradient_rows: Sequence[Mapping[str, Any]],
    stage_name: str,
    residual_opened: bool,
    snr_threshold: float = 2.0,
    n_rem_high_threshold: float = 1.0,
    useful_horizon_threshold: float = 1.0,
) -> dict[str, Any]:
    """Telemetry/terminal predicate for the opt-in conservative noise-floor stop."""

    policy_key = str(policy or "off").strip().lower() or "off"
    if policy_key not in {"off", "noise_floor_agreement_v1"}:
        raise ValueError("adapt_noise_floor_stop_policy must be one of {'off','noise_floor_agreement_v1'}.")
    snr_threshold_val = float(snr_threshold)
    n_rem_high_threshold_val = float(n_rem_high_threshold)
    useful_horizon_threshold_val = float(useful_horizon_threshold)
    for name, value in (
        ("adapt_noise_floor_snr_threshold", snr_threshold_val),
        ("adapt_noise_floor_n_rem_high_threshold", n_rem_high_threshold_val),
        ("adapt_noise_floor_useful_horizon_threshold", useful_horizon_threshold_val),
    ):
        if (not math.isfinite(float(value))) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")

    enabled = bool(policy_key == "noise_floor_agreement_v1")
    snapshot = _noise_floor_snapshot_dict(controller_snapshot)
    n_rem_high = _finite_float_or_none(snapshot.get("n_rem_high")) if snapshot is not None else None
    useful_horizon = _finite_float_or_none(snapshot.get("useful_horizon")) if snapshot is not None else None
    runway_fraction = _finite_float_or_none(snapshot.get("runway_fraction")) if snapshot is not None else None
    confidence_ratio = _finite_float_or_none(snapshot.get("confidence_ratio")) if snapshot is not None else None
    runway_gate = bool(
        snapshot is not None
        and n_rem_high is not None
        and useful_horizon is not None
        and runway_fraction is not None
        and confidence_ratio is not None
        and float(n_rem_high) <= float(n_rem_high_threshold_val)
        and float(useful_horizon) <= float(useful_horizon_threshold_val)
    )

    snr_rows = _noise_floor_gradient_snr_values(candidate_gradient_rows)
    max_snr = max((float(row["snr"]) for row in snr_rows), default=None)
    phase3_snr_gate = bool(max_snr is not None and float(max_snr) <= float(snr_threshold_val))
    residual_gate = bool(str(stage_name).strip().lower() == "residual" and bool(residual_opened))
    drop_gate = bool(drop_policy_enabled and drop_plateau_gate)
    missing: list[str] = []
    if snapshot is None:
        missing.append("controller_snapshot")
    else:
        if n_rem_high is None:
            missing.append("controller_snapshot.n_rem_high")
        if useful_horizon is None:
            missing.append("controller_snapshot.useful_horizon")
        if runway_fraction is None:
            missing.append("controller_snapshot.runway_fraction")
        if confidence_ratio is None:
            missing.append("controller_snapshot.confidence_ratio")
    if max_snr is None:
        missing.append("current_window_phase3_snr")
    terminal = bool(enabled and drop_gate and runway_gate and phase3_snr_gate and residual_gate)
    pre_residual_agreement = bool(enabled and drop_gate and runway_gate and phase3_snr_gate and not residual_gate)
    return {
        "schema_version": "noise_floor_agreement_v1",
        "policy": str(policy_key),
        "enabled": bool(enabled),
        "terminal_stop": bool(terminal),
        "pre_residual_agreement": bool(pre_residual_agreement),
        "missing_telemetry_reasons": [str(x) for x in missing],
        "drop_plateau_gate": bool(drop_gate),
        "drop_policy_enabled": bool(drop_policy_enabled),
        "depth_local": int(depth_local),
        "drop_plateau_hits": int(drop_plateau_hits),
        "adapt_drop_patience": int(adapt_drop_patience),
        "runway_gate": bool(runway_gate),
        "controller_snapshot_present": bool(snapshot is not None),
        "n_rem_high": n_rem_high,
        "n_rem_high_threshold": float(n_rem_high_threshold_val),
        "useful_horizon": useful_horizon,
        "useful_horizon_threshold": float(useful_horizon_threshold_val),
        "runway_fraction": runway_fraction,
        "confidence_ratio": confidence_ratio,
        "phase3_snr_gate": bool(phase3_snr_gate),
        "phase3_snr_threshold": float(snr_threshold_val),
        "phase3_snr_sample_count": int(len(snr_rows)),
        "phase3_snr_max": max_snr,
        "phase3_snr_rows_sample": [dict(row) for row in snr_rows[:8]],
        "residual_stage_agreement": bool(residual_gate),
        "stage_name": str(stage_name),
        "residual_opened": bool(residual_opened),
        "low_snr_alone_can_stop": False,
    }
