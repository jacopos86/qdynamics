from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from pipelines.exact_bench.noise_oracle_defaults import (
    SYNTHETIC_COHERENT_1Q_GATES_DEFAULT,
    SYNTHETIC_COHERENT_2Q_GATES_DEFAULT,
    SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT,
    SYNTHETIC_DEPOLARIZING_1Q_GATES_DEFAULT,
    SYNTHETIC_DEPOLARIZING_2Q_GATES_DEFAULT,
    normalize_gate_name_tuple,
)
from typing import Any, Callable, Sequence

from pipelines.static_adapt.builders.problem_registry import (
    available_adapt_pool_keys,
    available_problem_keys,
    supported_continuation_modes_for_problem,
)
from pipelines.static_adapt.builders.problem_setup import _HH_STAGED_CONTINUATION_MODES
from pipelines.static_adapt.engine_support import _resolve_cli_adapt_continuation_mode
from pipelines.static_adapt.accepted_refit import (
    ACCEPTED_REFIT_BASE_CHART_CHOICES,
    ACCEPTED_REFIT_CHART_CHOICES,
    ACCEPTED_REFIT_CHART_NATIVE_V1,
    ACCEPTED_REFIT_SCOPE_CHOICES,
    ACCEPTED_REFIT_SCOPE_SELECTOR_POLICY_V1,
)
from pipelines.static_adapt.plateau_acquisition import (
    PLATEAU_ACQUISITION_SCORE_CHOICES,
    PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
    PLATEAU_ACQUISITION_MODE_CHOICES,
    PLATEAU_ACQUISITION_MODE_OFF,
    PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
    PLATEAU_DUPLICATE_POLICY_CHOICES,
    PLATEAU_SEED_PROBE_MODE_CHOICES,
    PLATEAU_SEED_PROBE_MODE_OFF,
    PLATEAU_TRIAL_OPTIMIZER_CHOICES,
    PLATEAU_TRIAL_OPTIMIZER_INHERIT,
)
from pipelines.static_adapt.lane_routes import (
    PHYSICAL_LANE_SHORTLIST_AGGRESSIVENESS_CHOICES,
    STATIC_LANE_ROUTE_CHOICES,
    STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
)
from pipelines.static_adapt.route_a_child_padding import (
    ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1,
    RouteAChildPaddingConfig,
)
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_CONTROLLER_ABLATION_CONTRACT_CHOICES,
    SR_CONTROLLER_ABLATION_CONTRACT_OFF,
    SR_COORDINATE_SOLVE_SCOPE_CHOICES,
    SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
    SR_ESCAPE_DISABLED,
    SR_ESCAPE_MODE_CHOICES,
    SR_POWELL_COORDINATE_CHART_AUTO,
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
    SR_POWELL_COORDINATE_CHART_REQUEST_CHOICES,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    PHASE3_RESPONSE_COORDINATE_SCOPE_CHOICES,
    PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1,
    SR_ROUTE_PROFILE_REQUEST_CHOICES,
    SR_ROUTE_PROFILE_REQUEST_OFF,
    SR_ROUTE_PROFILE_CONVENTIONAL_V3,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
    normalize_sr_route_profile_request,
    normalize_sr_route_profile_namespace,
)
from pipelines.static_adapt.sr_snake_phase12_policy import (
    PHASE1_ENERGY_MODEL_CHOICES,
    PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_CHOICES,
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF,
    PHASE2_CURVATURE_POLICY_CHOICES,
    PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
)
from pipelines.static_adapt.route_a_schur_selector import (
    ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
    ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1,
    ROUTE_A_TRUST_REGION_FIXED,
)
from pipelines.static_adapt.paper_i_config import PAPER_I_CANONICAL_COST_WEIGHTS
from pipelines.scaffold.hh_continuation_pruning import (
    PRUNE_METRIC_COST_WEIGHT_ANSATZ_ENTRY_DENOMINATOR_V1,
    PRUNE_METRIC_COST_WEIGHT_OFF,
    PRUNE_METRIC_SCHUR_SOLVE_GRADIENT_CORRECTED_V1,
    PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
    PRUNE_SCHUR_ROUTE_HESSIAN_COUPLING_V1,
    PRUNE_SCHUR_ROUTE_METRIC_REGULARIZED_V1,
)
from pipelines.scaffold.hh_continuation_scoring import (
    GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1,
)

CANONICAL_HARDWARE_COST_LAMBDA_2Q = PAPER_I_CANONICAL_COST_WEIGHTS.lambda_2q
CANONICAL_HARDWARE_COST_LAMBDA_D = PAPER_I_CANONICAL_COST_WEIGHTS.lambda_d
CANONICAL_HARDWARE_COST_LAMBDA_1Q = PAPER_I_CANONICAL_COST_WEIGHTS.lambda_1q
CANONICAL_HARDWARE_COST_LAMBDA_THETA = PAPER_I_CANONICAL_COST_WEIGHTS.lambda_theta
CANONICAL_HARDWARE_COST_LAMBDA_SHOT = PAPER_I_CANONICAL_COST_WEIGHTS.lambda_shot

_DEFERRED_GRAM_FALLBACK_POLICY_CHOICES = (
    "off",
    GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1,
)
_VALUE_NOISE_MODE_CHOICES = ("off", "gaussian_iid_v1")
_VALUE_NOISE_SEMANTIC = "post_expectation_value_noise_not_physical_shots"
_SHOT_EQUIVALENT_VALUE_NOISE_SEMANTIC = "snake_function_value_noise_shot_equivalent_v1"
_VALUE_NOISE_STD_MATCH_RTOL = 1e-9
_VALUE_NOISE_STD_MATCH_ATOL = 1e-15
_ADAPT_NOISE_FLOOR_STOP_POLICY_CHOICES = ("off", "noise_floor_agreement_v1")
_PHASE3_ORACLE_GRADIENT_MODE_CHOICES = (
    "off",
    "ideal",
    "shots",
    "aer_noise",
    "aer_density_matrix",
    "aer_density_matrix_synthetic_depolarizing",
    "aer_density_matrix_synthetic_coherent",
    "backend_scheduled",
    "runtime",
)



@dataclass(frozen=True)
class Phase3OracleGradientConfig:
    noise_mode: str
    shots: int
    oracle_repeats: int
    oracle_aggregate: str
    backend_name: str | None
    use_fake_backend: bool
    seed: int
    gradient_step: float
    mitigation_mode: str
    local_readout_strategy: str | None
    zne_scales: tuple[float, ...] = ()
    local_gate_twirling: bool = False
    dd_sequence: str | None = None
    scope: str = "selection_only"
    execution_surface_requested: str = "auto"
    execution_surface: str = "expectation_v1"
    raw_transport: str = "auto"
    raw_store_memory: bool = False
    raw_artifact_path: str | None = None
    seed_transpiler: int | None = None
    transpile_optimization_level: int = 1
    value_noise_model: str = "off"
    value_noise_std: float = 0.0
    value_noise_seed: int | None = None
    value_noise_sigma0_abs: float | None = None
    value_noise_n_eff: float | None = None
    value_noise_semantic: str = _VALUE_NOISE_SEMANTIC
    value_noise_std_source: str = "explicit_std"
    value_noise_physical_shots_unchanged: bool = True
    value_noise_fixed_gate_error_reduction_claimed: bool = False
    synthetic_depolarizing_1q_error: float = 0.0
    synthetic_depolarizing_2q_error: float = 0.0
    synthetic_depolarizing_1q_gates: tuple[str, ...] = SYNTHETIC_DEPOLARIZING_1Q_GATES_DEFAULT
    synthetic_depolarizing_2q_gates: tuple[str, ...] = SYNTHETIC_DEPOLARIZING_2Q_GATES_DEFAULT
    synthetic_coherent_1q_angle_std: float = 0.0
    synthetic_coherent_2q_angle_std: float = 0.0
    synthetic_coherent_seed: int | None = None
    synthetic_coherent_generator_mode: str = SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT
    synthetic_coherent_1q_gates: tuple[str, ...] = SYNTHETIC_COHERENT_1Q_GATES_DEFAULT
    synthetic_coherent_2q_gates: tuple[str, ...] = SYNTHETIC_COHERENT_2Q_GATES_DEFAULT

_FINAL_NOISE_AUDIT_RUNTIME_PROFILE_NAMES = {
    "legacy_runtime_v0",
    "main_twirled_readout_v1",
    "dd_probe_twirled_readout_v1",
    "final_audit_zne_twirled_readout_v1",
}

_FINAL_NOISE_AUDIT_RUNTIME_SESSION_POLICIES = {
    "prefer_session",
    "require_session",
    "backend_only",
}


@dataclass(frozen=True)
class FinalNoiseAuditConfig:
    noise_mode: str
    shots: int
    oracle_repeats: int
    oracle_aggregate: str
    backend_name: str | None
    use_fake_backend: bool
    seed: int
    mitigation_mode: str
    local_readout_strategy: str | None
    zne_scales: tuple[float, ...] = ()
    local_gate_twirling: bool = False
    dd_sequence: str | None = None
    runtime_profile_name: str = "legacy_runtime_v0"
    runtime_session_policy: str = "prefer_session"
    compare_unmitigated_baseline: bool = False
    seed_transpiler: int | None = None
    transpile_optimization_level: int = 1
    strict: bool = False
    value_noise_model: str = "off"
    value_noise_std: float = 0.0
    value_noise_seed: int | None = None


def _parse_oracle_zne_scales(
    raw: Any,
    *,
    field_name: str,
) -> tuple[float, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        tokens = [tok.strip() for tok in str(raw).split(",") if tok.strip() != ""]
    elif isinstance(raw, Sequence):
        tokens = list(raw)
    else:
        tokens = [raw]
    out: list[float] = []
    for tok in tokens:
        value = float(tok)
        if (not math.isfinite(value)) or value <= 0.0:
            raise ValueError(f"{field_name} entries must be finite and > 0.")
        out.append(float(value))
    return tuple(out)

def _parse_gate_name_tuple(raw: Any, *, default: Sequence[str], field_name: str) -> tuple[str, ...]:
    return normalize_gate_name_tuple(raw, default=default, field_name=field_name)


def _resolve_value_noise_model(raw: Any) -> str:
    model = str(raw if raw is not None else "off").strip().lower() or "off"
    if model not in _VALUE_NOISE_MODE_CHOICES:
        raise ValueError(
            f"value_noise_model must be one of {set(_VALUE_NOISE_MODE_CHOICES)}."
        )
    return str(model)


def _resolve_value_noise_seed(raw: Any) -> int | None:
    if raw in {None, ""}:
        return None
    return int(raw)


def _resolve_value_noise_std_contract(
    *,
    label: str,
    value_noise_model: str,
    value_noise_std: Any = None,
    value_noise_sigma0_abs: Any = None,
    value_noise_n_eff: Any = None,
    std_match_rtol: float = _VALUE_NOISE_STD_MATCH_RTOL,
    std_match_atol: float = _VALUE_NOISE_STD_MATCH_ATOL,
) -> tuple[float, dict[str, Any]]:
    """Resolve explicit or shot-equivalent value-noise std plus audit metadata."""

    model = _resolve_value_noise_model(value_noise_model)
    std_provided = value_noise_std not in {None, ""}
    sigma0_provided = value_noise_sigma0_abs not in {None, ""}
    n_eff_provided = value_noise_n_eff not in {None, ""}
    if sigma0_provided != n_eff_provided:
        raise ValueError(
            f"{label}_value_noise_sigma0_abs and {label}_value_noise_n_eff must be supplied together."
        )

    explicit_std = 0.0 if not std_provided else float(value_noise_std)
    if std_provided and ((not math.isfinite(explicit_std)) or explicit_std < 0.0):
        raise ValueError(f"{label}_value_noise_std must be finite and nonnegative.")

    sigma0_abs: float | None = None
    n_eff: float | None = None
    derived_std: float | None = None
    std_source = "explicit_std" if std_provided else "default_zero"
    semantic = _VALUE_NOISE_SEMANTIC

    if sigma0_provided:
        sigma0_abs = float(value_noise_sigma0_abs)
        n_eff = float(value_noise_n_eff)
        if (not math.isfinite(sigma0_abs)) or sigma0_abs <= 0.0:
            raise ValueError(f"{label}_value_noise_sigma0_abs must be finite and > 0.")
        if (not math.isfinite(n_eff)) or n_eff <= 0.0:
            raise ValueError(f"{label}_value_noise_n_eff must be finite and > 0.")
        derived_std = float(sigma0_abs / math.sqrt(n_eff))
        if model == "off":
            raise ValueError(
                f"{label}_value_noise_sigma0_abs/{label}_value_noise_n_eff require "
                f"{label}_value_noise_model='gaussian_iid_v1'."
            )
        if std_provided and not math.isclose(
            float(explicit_std),
            float(derived_std),
            rel_tol=float(std_match_rtol),
            abs_tol=float(std_match_atol),
        ):
            raise ValueError(
                f"{label}_value_noise_std mismatch: explicit {explicit_std} does not match "
                f"sigma0_abs/sqrt(N_eff)={derived_std}."
            )
        explicit_std = float(derived_std)
        std_source = "sigma0_abs_over_sqrt_N_eff"
        semantic = _SHOT_EQUIVALENT_VALUE_NOISE_SEMANTIC

    payload = {
        "enabled": bool(model != "off"),
        "model": str(model),
        "std": float(explicit_std),
        "explicit_std": float(value_noise_std) if std_provided else None,
        "sigma0_abs": sigma0_abs,
        "N_eff": n_eff,
        "derived_std": derived_std,
        "std_source": str(std_source),
        "std_match_rtol": float(std_match_rtol),
        "std_match_atol": float(std_match_atol),
        "semantic": str(semantic),
        "physical_shots_unchanged": True,
        "fixed_gate_error_reduction_claimed": False,
    }
    return float(explicit_std), payload


def _validate_value_noise_config(
    *,
    label: str,
    value_noise_model: str,
    value_noise_std: float | None,
    execution_surface: str = "expectation_v1",
) -> None:
    model = _resolve_value_noise_model(value_noise_model)
    std = 0.0 if value_noise_std in {None, ""} else float(value_noise_std)
    if model == "off":
        if std != 0.0:
            raise ValueError(f"{label}_value_noise_model='off' requires {label}_value_noise_std == 0.")
        return
    if model == "gaussian_iid_v1":
        if (not math.isfinite(std)) or std <= 0.0:
            raise ValueError(
                f"{label}_value_noise_model='gaussian_iid_v1' requires finite {label}_value_noise_std > 0."
            )
        if str(execution_surface).strip().lower() != "expectation_v1":
            raise ValueError(
                f"{label} value noise requires execution_surface='expectation_v1'; raw_measurement_v1 is unsupported."
            )
        return
    raise ValueError(f"Unsupported {label}_value_noise_model {model!r}.")


def _value_noise_payload_from_fields(
    *,
    value_noise_model: str,
    value_noise_std: float,
    value_noise_seed: int | None,
    value_noise_sigma0_abs: float | None = None,
    value_noise_n_eff: float | None = None,
    value_noise_semantic: str | None = None,
    value_noise_std_source: str | None = None,
) -> dict[str, Any]:
    model = _resolve_value_noise_model(value_noise_model)
    semantic = (
        str(value_noise_semantic)
        if value_noise_semantic not in {None, ""}
        else _VALUE_NOISE_SEMANTIC
    )
    derived_std = (
        None
        if value_noise_sigma0_abs is None or value_noise_n_eff is None
        else float(value_noise_sigma0_abs) / math.sqrt(float(value_noise_n_eff))
    )
    return {
        "enabled": bool(model != "off"),
        "model": str(model),
        "std": float(value_noise_std),
        "seed": (None if value_noise_seed is None else int(value_noise_seed)),
        "semantic": str(semantic),
        "sigma0_abs": None if value_noise_sigma0_abs is None else float(value_noise_sigma0_abs),
        "N_eff": None if value_noise_n_eff is None else float(value_noise_n_eff),
        "derived_std": derived_std,
        "std_source": (
            str(value_noise_std_source)
            if value_noise_std_source not in {None, ""}
            else ("sigma0_abs_over_sqrt_N_eff" if derived_std is not None else "explicit_std")
        ),
        "physical_shots_unchanged": True,
        "fixed_gate_error_reduction_claimed": False,
    }


def _validate_backend_scheduled_local_zne_scales(
    zne_scales: Sequence[float],
    *,
    field_name: str,
) -> tuple[float, ...]:
    out: list[float] = []
    for raw in zne_scales:
        value = float(raw)
        rounded = int(round(value))
        if (
            (not math.isfinite(value))
            or rounded < 1
            or rounded % 2 == 0
            or (not math.isclose(value, float(rounded), rel_tol=0.0, abs_tol=1e-9))
        ):
            raise ValueError(
                f"{field_name} must contain odd positive integer noise scales for backend_scheduled local ZNE."
            )
        out.append(float(rounded))
    if out and not any(math.isclose(val, 1.0, rel_tol=0.0, abs_tol=1e-9) for val in out):
        raise ValueError(
            f"{field_name} must include the base noise scale 1 for backend_scheduled local ZNE."
        )
    return tuple(out)

def _resolve_phase3_oracle_gradient_config(
    config: Phase3OracleGradientConfig,
) -> Phase3OracleGradientConfig:
    requested_surface = str(getattr(config, "execution_surface_requested", "auto")).strip().lower() or "auto"
    if requested_surface not in {"auto", "expectation_v1", "raw_measurement_v1"}:
        raise ValueError(
            "phase3_oracle_execution_surface must be one of {'auto','expectation_v1','raw_measurement_v1'}."
        )
    noise_mode = str(config.noise_mode).strip().lower()
    mitigation_mode = str(config.mitigation_mode).strip().lower()
    resolved_surface = (
        "raw_measurement_v1"
        if requested_surface == "auto"
        and noise_mode == "runtime"
        and mitigation_mode == "none"
        else (
            "expectation_v1"
            if requested_surface == "auto"
            else str(requested_surface)
        )
    )
    value_noise_model = _resolve_value_noise_model(getattr(config, "value_noise_model", "off"))
    value_noise_std, value_noise_contract = _resolve_value_noise_std_contract(
        label="phase3_oracle",
        value_noise_model=value_noise_model,
        value_noise_std=getattr(config, "value_noise_std", None),
        value_noise_sigma0_abs=getattr(config, "value_noise_sigma0_abs", None),
        value_noise_n_eff=getattr(config, "value_noise_n_eff", None),
    )
    synthetic_1q_gates = _parse_gate_name_tuple(
        getattr(config, "synthetic_depolarizing_1q_gates", None),
        default=SYNTHETIC_DEPOLARIZING_1Q_GATES_DEFAULT,
        field_name="phase3_oracle_synthetic_depolarizing_1q_gates",
    )
    synthetic_2q_gates = _parse_gate_name_tuple(
        getattr(config, "synthetic_depolarizing_2q_gates", None),
        default=SYNTHETIC_DEPOLARIZING_2Q_GATES_DEFAULT,
        field_name="phase3_oracle_synthetic_depolarizing_2q_gates",
    )
    coherent_1q_gates = _parse_gate_name_tuple(
        getattr(config, "synthetic_coherent_1q_gates", None),
        default=SYNTHETIC_COHERENT_1Q_GATES_DEFAULT,
        field_name="phase3_oracle_synthetic_coherent_1q_gates",
    )
    coherent_2q_gates = _parse_gate_name_tuple(
        getattr(config, "synthetic_coherent_2q_gates", None),
        default=SYNTHETIC_COHERENT_2Q_GATES_DEFAULT,
        field_name="phase3_oracle_synthetic_coherent_2q_gates",
    )
    return Phase3OracleGradientConfig(
        noise_mode=str(noise_mode),
        shots=int(config.shots),
        oracle_repeats=int(config.oracle_repeats),
        oracle_aggregate=str(config.oracle_aggregate).strip().lower(),
        backend_name=(None if config.backend_name in {None, ""} else str(config.backend_name)),
        use_fake_backend=bool(config.use_fake_backend),
        seed=int(config.seed),
        gradient_step=float(config.gradient_step),
        mitigation_mode=str(mitigation_mode),
        local_readout_strategy=(
            None
            if config.local_readout_strategy in {None, ""}
            else str(config.local_readout_strategy).strip().lower()
        ),
        zne_scales=_parse_oracle_zne_scales(
            getattr(config, "zne_scales", ()),
            field_name="phase3_oracle_zne_scales",
        ),
        local_gate_twirling=bool(getattr(config, "local_gate_twirling", False)),
        dd_sequence=(
            None
            if getattr(config, "dd_sequence", None) in {None, "", "none"}
            else str(getattr(config, "dd_sequence")).strip()
        ),
        scope=str(config.scope).strip().lower() or "selection_only",
        execution_surface_requested=str(requested_surface),
        execution_surface=str(resolved_surface),
        raw_transport=str(getattr(config, "raw_transport", "auto")).strip().lower() or "auto",
        raw_store_memory=bool(getattr(config, "raw_store_memory", False)),
        raw_artifact_path=(
            None
            if getattr(config, "raw_artifact_path", None) in {None, ""}
            else str(getattr(config, "raw_artifact_path"))
        ),
        seed_transpiler=(
            None
            if getattr(config, "seed_transpiler", None) is None
            else int(getattr(config, "seed_transpiler"))
        ),
        transpile_optimization_level=int(getattr(config, "transpile_optimization_level", 1)),
        value_noise_model=value_noise_model,
        value_noise_std=float(value_noise_std),
        value_noise_seed=_resolve_value_noise_seed(getattr(config, "value_noise_seed", None)),
        value_noise_sigma0_abs=value_noise_contract["sigma0_abs"],
        value_noise_n_eff=value_noise_contract["N_eff"],
        value_noise_semantic=str(value_noise_contract["semantic"]),
        value_noise_std_source=str(value_noise_contract["std_source"]),
        value_noise_physical_shots_unchanged=True,
        value_noise_fixed_gate_error_reduction_claimed=False,
        synthetic_depolarizing_1q_error=float(getattr(config, "synthetic_depolarizing_1q_error", 0.0)),
        synthetic_depolarizing_2q_error=float(getattr(config, "synthetic_depolarizing_2q_error", 0.0)),
        synthetic_depolarizing_1q_gates=synthetic_1q_gates,
        synthetic_depolarizing_2q_gates=synthetic_2q_gates,
        synthetic_coherent_1q_angle_std=float(getattr(config, "synthetic_coherent_1q_angle_std", 0.0)),
        synthetic_coherent_2q_angle_std=float(getattr(config, "synthetic_coherent_2q_angle_std", 0.0)),
        synthetic_coherent_seed=(
            None
            if getattr(config, "synthetic_coherent_seed", None) is None
            else int(getattr(config, "synthetic_coherent_seed"))
        ),
        synthetic_coherent_generator_mode=str(
            getattr(config, "synthetic_coherent_generator_mode", SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT)
        ).strip().lower(),
        synthetic_coherent_1q_gates=coherent_1q_gates,
        synthetic_coherent_2q_gates=coherent_2q_gates,
    )

def _resolve_final_noise_audit_config(
    config: FinalNoiseAuditConfig,
) -> FinalNoiseAuditConfig:
    return FinalNoiseAuditConfig(
        noise_mode=str(config.noise_mode).strip().lower(),
        shots=int(config.shots),
        oracle_repeats=int(config.oracle_repeats),
        oracle_aggregate=str(config.oracle_aggregate).strip().lower(),
        backend_name=(None if config.backend_name in {None, ""} else str(config.backend_name)),
        use_fake_backend=bool(config.use_fake_backend),
        seed=int(config.seed),
        mitigation_mode=str(config.mitigation_mode).strip().lower(),
        local_readout_strategy=(
            None
            if config.local_readout_strategy in {None, ""}
            else str(config.local_readout_strategy).strip().lower()
        ),
        zne_scales=_parse_oracle_zne_scales(
            getattr(config, "zne_scales", ()),
            field_name="final_noise_audit_zne_scales",
        ),
        local_gate_twirling=bool(getattr(config, "local_gate_twirling", False)),
        dd_sequence=(
            None
            if getattr(config, "dd_sequence", None) in {None, "", "none"}
            else str(getattr(config, "dd_sequence")).strip()
        ),
        runtime_profile_name=(
            str(getattr(config, "runtime_profile_name", "legacy_runtime_v0")).strip().lower()
            or "legacy_runtime_v0"
        ),
        runtime_session_policy=(
            str(getattr(config, "runtime_session_policy", "prefer_session")).strip().lower()
            or "prefer_session"
        ),
        compare_unmitigated_baseline=bool(
            getattr(config, "compare_unmitigated_baseline", False)
        ),
        seed_transpiler=(
            None if config.seed_transpiler is None else int(config.seed_transpiler)
        ),
        transpile_optimization_level=int(config.transpile_optimization_level),
        strict=bool(config.strict),
        value_noise_model=_resolve_value_noise_model(getattr(config, "value_noise_model", "off")),
        value_noise_std=float(getattr(config, "value_noise_std", 0.0)),
        value_noise_seed=_resolve_value_noise_seed(getattr(config, "value_noise_seed", None)),
    )

def _validate_phase3_oracle_gradient_config(
    *,
    config: Phase3OracleGradientConfig,
    problem: str,
    continuation_mode: str,
) -> None:
    config = _resolve_phase3_oracle_gradient_config(config)
    problem_key = str(problem).strip().lower()
    continuation_key = str(continuation_mode).strip().lower()
    if problem_key not in {"hh", "spin_boson", "hubbard"}:
        raise ValueError("phase3 oracle gradient mode is only valid for problem in {'hh','spin_boson','hubbard'}.")
    if continuation_key != "phase3_v1":
        raise ValueError(
            "phase3 oracle gradient mode is only valid for adapt_continuation_mode='phase3_v1'."
        )
    noise_mode = str(config.noise_mode).strip().lower()
    if noise_mode not in set(_PHASE3_ORACLE_GRADIENT_MODE_CHOICES) - {"off"}:
        raise ValueError(
            "phase3_oracle_gradient_mode must be one of "
            f"{tuple(mode for mode in _PHASE3_ORACLE_GRADIENT_MODE_CHOICES if mode != 'off')}."
        )
    if int(config.shots) < 1:
        raise ValueError("phase3_oracle_shots must be >= 1.")
    if int(config.oracle_repeats) < 1:
        raise ValueError("phase3_oracle_repeats must be >= 1.")
    aggregate_key = str(config.oracle_aggregate).strip().lower()
    if aggregate_key != "mean":
        raise ValueError("phase3 oracle gradient mode currently requires oracle_aggregate='mean'.")
    if (not math.isfinite(float(config.gradient_step))) or float(config.gradient_step) <= 0.0:
        raise ValueError("phase3_oracle_gradient_step must be finite and > 0.")
    mitigation_mode = str(config.mitigation_mode).strip().lower()
    if mitigation_mode not in {"none", "readout"}:
        raise ValueError("phase3_oracle_mitigation must be one of {'none','readout'}.")
    zne_scales = tuple(float(x) for x in getattr(config, "zne_scales", ()) or ())
    local_gate_twirling = bool(getattr(config, "local_gate_twirling", False))
    dd_sequence = (
        None
        if getattr(config, "dd_sequence", None) in {None, "", "none"}
        else str(getattr(config, "dd_sequence")).strip()
    )
    if noise_mode == "backend_scheduled" and not bool(config.use_fake_backend):
        raise ValueError(
            "phase3 oracle gradient mode backend_scheduled requires --phase3-oracle-use-fake-backend."
        )
    if noise_mode == "runtime" and config.backend_name in {None, ""}:
        raise ValueError("phase3 oracle gradient runtime mode requires --phase3-oracle-backend-name.")
    local_readout_strategy = (
        None
        if config.local_readout_strategy in {None, ""}
        else str(config.local_readout_strategy).strip().lower()
    )
    if noise_mode == "aer_density_matrix" and mitigation_mode != "none":
        raise ValueError(
            "phase3 oracle aer_density_matrix mode requires phase3_oracle_mitigation='none'."
        )
    if mitigation_mode == "readout" and local_readout_strategy not in {None, "mthree"}:
        raise ValueError(
            "phase3_oracle_local_readout_strategy must be 'mthree' when readout mitigation is enabled."
        )
    if mitigation_mode != "readout" and local_readout_strategy is not None:
        raise ValueError(
            "phase3_oracle_local_readout_strategy requires phase3_oracle_mitigation='readout'."
        )
    if noise_mode != "backend_scheduled" and (
        zne_scales
        or local_gate_twirling
        or dd_sequence not in {None, "", "none"}
    ):
        raise ValueError(
            "phase3 oracle local ZNE/gate twirling/DD currently require noise_mode='backend_scheduled'."
        )
    execution_surface = str(config.execution_surface).strip().lower()
    synthetic_p1q = float(getattr(config, "synthetic_depolarizing_1q_error", 0.0))
    synthetic_p2q = float(getattr(config, "synthetic_depolarizing_2q_error", 0.0))
    synthetic_mode = noise_mode == "aer_density_matrix_synthetic_depolarizing"
    synthetic_depolarizing_channel_mode = noise_mode in {
        "aer_density_matrix_synthetic_depolarizing",
        "aer_density_matrix_synthetic_coherent",
    }
    coherent_s1q = float(getattr(config, "synthetic_coherent_1q_angle_std", 0.0))
    coherent_s2q = float(getattr(config, "synthetic_coherent_2q_angle_std", 0.0))
    coherent_mode = noise_mode == "aer_density_matrix_synthetic_coherent"
    if (not math.isfinite(synthetic_p1q)) or (not math.isfinite(synthetic_p2q)):
        raise ValueError("phase3 oracle synthetic depolarizing errors must be finite.")
    if synthetic_p1q < 0.0 or synthetic_p1q > 1.0 or synthetic_p2q < 0.0 or synthetic_p2q > 1.0:
        raise ValueError("phase3 oracle synthetic depolarizing errors must satisfy 0 <= p <= 1.")
    if not synthetic_depolarizing_channel_mode and (synthetic_p1q != 0.0 or synthetic_p2q != 0.0):
        raise ValueError(
            "phase3 oracle synthetic depolarizing errors require noise_mode='aer_density_matrix_synthetic_depolarizing' or 'aer_density_matrix_synthetic_coherent'."
        )
    if synthetic_mode:
        if bool(getattr(config, "use_fake_backend", False)):
            raise ValueError("aer_density_matrix_synthetic_depolarizing does not use fake-backend hardware metadata.")
        if str(getattr(config, "backend_name", "") or "").strip():
            raise ValueError("aer_density_matrix_synthetic_depolarizing must not set phase3_oracle_backend_name.")
        if execution_surface != "expectation_v1":
            raise ValueError("aer_density_matrix_synthetic_depolarizing requires execution_surface='expectation_v1'.")
        if mitigation_mode != "none":
            raise ValueError("aer_density_matrix_synthetic_depolarizing requires phase3_oracle_mitigation='none'.")
    if synthetic_depolarizing_channel_mode:
        if synthetic_p1q > 0.0 and not tuple(getattr(config, "synthetic_depolarizing_1q_gates", ()) or ()):
            raise ValueError("phase3_oracle_synthetic_depolarizing_1q_gates must be non-empty when p1q > 0.")
        if synthetic_p2q > 0.0 and not tuple(getattr(config, "synthetic_depolarizing_2q_gates", ()) or ()):
            raise ValueError("phase3_oracle_synthetic_depolarizing_2q_gates must be non-empty when p2q > 0.")
    if (not math.isfinite(coherent_s1q)) or (not math.isfinite(coherent_s2q)):
        raise ValueError("phase3 oracle synthetic coherent angle std values must be finite.")
    if coherent_s1q < 0.0 or coherent_s2q < 0.0:
        raise ValueError("phase3 oracle synthetic coherent angle std values must be nonnegative.")
    if not coherent_mode and (coherent_s1q != 0.0 or coherent_s2q != 0.0):
        raise ValueError(
            "phase3 oracle synthetic coherent angle std requires noise_mode='aer_density_matrix_synthetic_coherent'."
        )
    if coherent_mode:
        if bool(getattr(config, "use_fake_backend", False)):
            raise ValueError("aer_density_matrix_synthetic_coherent does not use fake-backend hardware metadata.")
        if str(getattr(config, "backend_name", "") or "").strip():
            raise ValueError("aer_density_matrix_synthetic_coherent must not set phase3_oracle_backend_name.")
        if execution_surface != "expectation_v1":
            raise ValueError("aer_density_matrix_synthetic_coherent requires execution_surface='expectation_v1'.")
        if mitigation_mode != "none":
            raise ValueError("aer_density_matrix_synthetic_coherent requires phase3_oracle_mitigation='none'.")
        if coherent_s1q > 0.0 and not tuple(getattr(config, "synthetic_coherent_1q_gates", ()) or ()):
            raise ValueError("phase3_oracle_synthetic_coherent_1q_gates must be non-empty when 1q angle std > 0.")
        if coherent_s2q > 0.0 and not tuple(getattr(config, "synthetic_coherent_2q_gates", ()) or ()):
            raise ValueError("phase3_oracle_synthetic_coherent_2q_gates must be non-empty when 2q angle std > 0.")
        if (
            str(getattr(config, "synthetic_coherent_generator_mode", SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT))
            .strip()
            .lower()
            != SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT
        ):
            raise ValueError(
                "phase3_oracle_synthetic_coherent_generator_mode must be "
                f"{SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT!r}."
            )
    if noise_mode == "backend_scheduled" and zne_scales:
        _validate_backend_scheduled_local_zne_scales(
            zne_scales,
            field_name="phase3_oracle_zne_scales",
        )
    if str(config.scope).strip().lower() != "selection_only":
        raise ValueError("phase3 oracle gradient scope is fixed to 'selection_only' in v1.")
    _validate_value_noise_config(
        label="phase3_oracle",
        value_noise_model=str(getattr(config, "value_noise_model", "off")),
        value_noise_std=float(getattr(config, "value_noise_std", 0.0)),
        execution_surface=str(execution_surface),
    )
    if problem_key == "spin_boson":
        if noise_mode != "ideal":
            raise ValueError("spin_boson phase3 oracle gradient mode currently supports only noise_mode='ideal'.")
        if bool(getattr(config, "use_fake_backend", False)):
            raise ValueError("spin_boson phase3 oracle gradient mode must not use a fake backend.")
        if str(getattr(config, "backend_name", "") or "").strip():
            raise ValueError("spin_boson phase3 oracle gradient mode must not set phase3_oracle_backend_name.")
        if execution_surface != "expectation_v1":
            raise ValueError("spin_boson phase3 oracle gradient mode requires execution_surface='expectation_v1'.")
        if mitigation_mode != "none":
            raise ValueError("spin_boson phase3 oracle gradient mode requires phase3_oracle_mitigation='none'.")
        if local_readout_strategy is not None:
            raise ValueError("spin_boson phase3 oracle gradient mode does not allow local readout mitigation.")
        if zne_scales:
            raise ValueError("spin_boson phase3 oracle gradient mode does not allow local ZNE scales.")
        if local_gate_twirling:
            raise ValueError("spin_boson phase3 oracle gradient mode does not allow local gate twirling.")
        if dd_sequence not in {None, "", "none"}:
            raise ValueError("spin_boson phase3 oracle gradient mode does not allow DD.")
        if synthetic_p1q != 0.0 or synthetic_p2q != 0.0:
            raise ValueError("spin_boson phase3 oracle gradient mode requires zero synthetic depolarizing errors.")
        if coherent_s1q != 0.0 or coherent_s2q != 0.0:
            raise ValueError("spin_boson phase3 oracle gradient mode requires zero synthetic coherent angle std.")
        return
    if problem_key == "hubbard":
        if noise_mode not in {"ideal", "aer_density_matrix_synthetic_depolarizing", "aer_density_matrix_synthetic_coherent"}:
            raise ValueError(
                "hubbard phase3 oracle gradient mode currently supports only noise_mode in "
                "{'ideal','aer_density_matrix_synthetic_depolarizing','aer_density_matrix_synthetic_coherent'}."
            )
        if bool(getattr(config, "use_fake_backend", False)):
            raise ValueError("hubbard phase3 oracle gradient mode must not use a fake backend.")
        if str(getattr(config, "backend_name", "") or "").strip():
            raise ValueError("hubbard phase3 oracle gradient mode must not set phase3_oracle_backend_name.")
        if execution_surface != "expectation_v1":
            raise ValueError("hubbard phase3 oracle gradient mode requires execution_surface='expectation_v1'.")
        if mitigation_mode != "none":
            raise ValueError("hubbard phase3 oracle gradient mode requires phase3_oracle_mitigation='none'.")
        if local_readout_strategy is not None:
            raise ValueError("hubbard phase3 oracle gradient mode does not allow local readout mitigation.")
        if zne_scales:
            raise ValueError("hubbard phase3 oracle gradient mode does not allow local ZNE scales.")
        if local_gate_twirling:
            raise ValueError("hubbard phase3 oracle gradient mode does not allow local gate twirling.")
        if dd_sequence not in {None, "", "none"}:
            raise ValueError("hubbard phase3 oracle gradient mode does not allow DD.")
        return
    if execution_surface == "expectation_v1":
        return
    if execution_surface != "raw_measurement_v1":
        raise ValueError(
            "phase3_oracle_execution_surface must resolve to 'expectation_v1' or 'raw_measurement_v1'."
        )
    if mitigation_mode != "none":
        raise ValueError("phase3 raw oracle execution requires mitigation_mode='none'.")
    if local_readout_strategy is not None:
        raise ValueError("phase3 raw oracle execution does not allow local readout strategy.")
    if zne_scales:
        raise ValueError("phase3 raw oracle execution does not allow local ZNE scales.")
    if local_gate_twirling:
        raise ValueError("phase3 raw oracle execution does not allow local gate twirling.")
    if dd_sequence not in {None, "", "none"}:
        raise ValueError("phase3 raw oracle execution does not allow local DD.")
    raw_transport = str(config.raw_transport).strip().lower()
    if noise_mode == "backend_scheduled":
        if not bool(config.use_fake_backend):
            raise ValueError(
                "phase3 raw oracle execution requires --phase3-oracle-use-fake-backend when noise_mode='backend_scheduled'."
            )
        if raw_transport != "auto":
            raise ValueError(
                "phase3 backend_scheduled raw oracle execution currently requires phase3_oracle_raw_transport='auto'."
            )
    else:
        if noise_mode != "runtime":
            raise ValueError(
                "phase3 raw oracle execution currently supports only noise_mode in {'runtime','backend_scheduled'}."
            )
        if bool(config.use_fake_backend):
            raise ValueError(
                "phase3 raw oracle execution requires a real runtime backend when noise_mode='runtime'."
            )
        if raw_transport not in {"auto", "sampler_v2"}:
            raise ValueError(
                "phase3_oracle_raw_transport must be one of {'auto','sampler_v2'}."
            )
    if int(config.transpile_optimization_level) not in {0, 1, 2, 3}:
        raise ValueError(
            "phase3_oracle_transpile_optimization_level must be one of {0,1,2,3}."
        )

def _oracle_mitigation_payload_from_fields(
    *,
    mitigation_mode: str,
    local_readout_strategy: str | None,
    zne_scales: Sequence[float] = (),
    dd_sequence: str | None = None,
    local_gate_twirling: bool = False,
) -> dict[str, Any]:
    mitigation_mode_key = str(mitigation_mode).strip().lower()
    local_readout_strategy_key = (
        "mthree"
        if mitigation_mode_key == "readout" and local_readout_strategy in {None, ""}
        else (
            None
            if local_readout_strategy in {None, ""}
            else str(local_readout_strategy).strip().lower()
        )
    )
    payload = {
        "mode": str(mitigation_mode_key),
        "zne_scales": [float(x) for x in zne_scales],
        "dd_sequence": (
            None
            if dd_sequence in {None, "", "none"}
            else str(dd_sequence).strip()
        ),
        "local_readout_strategy": local_readout_strategy_key,
    }
    if bool(local_gate_twirling):
        payload["local_gate_twirling"] = True
        payload["local_gate_twirling_scope"] = "2q_only"
    return payload

def _validate_final_noise_audit_config(
    *,
    config: FinalNoiseAuditConfig,
    problem: str,
) -> None:
    config = _resolve_final_noise_audit_config(config)
    problem_key = str(problem).strip().lower()
    if problem_key != "hh":
        raise ValueError("final noise audit is currently only valid for problem='hh'.")
    noise_mode = str(config.noise_mode).strip().lower()
    if noise_mode not in {"ideal", "shots", "aer_noise", "aer_density_matrix", "backend_scheduled", "runtime"}:
        raise ValueError(
            "final_noise_audit_mode must be one of {'off','ideal','shots','aer_noise','aer_density_matrix','backend_scheduled','runtime'}."
        )
    if int(config.shots) < 1:
        raise ValueError("final_noise_audit_shots must be >= 1.")
    if int(config.oracle_repeats) < 1:
        raise ValueError("final_noise_audit_repeats must be >= 1.")
    if str(config.oracle_aggregate) != "mean":
        raise ValueError("final noise audit currently requires oracle_aggregate='mean'.")
    if int(config.transpile_optimization_level) not in {0, 1, 2, 3}:
        raise ValueError(
            "final_noise_audit_transpile_optimization_level must be one of {0,1,2,3}."
        )
    _validate_value_noise_config(
        label="final_noise_audit",
        value_noise_model=str(getattr(config, "value_noise_model", "off")),
        value_noise_std=float(getattr(config, "value_noise_std", 0.0)),
        execution_surface="expectation_v1",
    )
    mitigation_mode = str(config.mitigation_mode).strip().lower()
    if mitigation_mode not in {"none", "readout"}:
        raise ValueError("final_noise_audit_mitigation must be one of {'none','readout'}.")
    zne_scales = tuple(float(x) for x in getattr(config, "zne_scales", ()) or ())
    local_gate_twirling = bool(getattr(config, "local_gate_twirling", False))
    dd_sequence = (
        None
        if getattr(config, "dd_sequence", None) in {None, "", "none"}
        else str(getattr(config, "dd_sequence")).strip()
    )
    runtime_profile_name = str(config.runtime_profile_name)
    runtime_session_policy = str(config.runtime_session_policy)
    if runtime_profile_name not in _FINAL_NOISE_AUDIT_RUNTIME_PROFILE_NAMES:
        raise ValueError(
            "final_noise_audit_runtime_profile must be one of "
            f"{sorted(_FINAL_NOISE_AUDIT_RUNTIME_PROFILE_NAMES)}."
        )
    if runtime_session_policy not in _FINAL_NOISE_AUDIT_RUNTIME_SESSION_POLICIES:
        raise ValueError(
            "final_noise_audit_runtime_session_policy must be one of "
            f"{sorted(_FINAL_NOISE_AUDIT_RUNTIME_SESSION_POLICIES)}."
        )
    local_readout_strategy = (
        None
        if config.local_readout_strategy in {None, ""}
        else str(config.local_readout_strategy)
    )
    if noise_mode == "aer_density_matrix" and mitigation_mode != "none":
        raise ValueError(
            "final noise audit aer_density_matrix mode requires final_noise_audit_mitigation='none'."
        )
    if mitigation_mode == "readout":
        if noise_mode == "backend_scheduled":
            if local_readout_strategy not in {None, "mthree"}:
                raise ValueError(
                    "final_noise_audit_local_readout_strategy must be 'mthree' when backend_scheduled readout mitigation is enabled."
                )
        elif noise_mode == "runtime":
            if local_readout_strategy is not None:
                raise ValueError(
                    "final noise audit runtime readout uses provider-side mitigation and does not accept local readout strategy."
                )
        else:
            raise ValueError(
                "final noise audit readout mitigation is currently supported only for noise_mode in {'backend_scheduled','runtime'}."
            )
    elif local_readout_strategy is not None:
        raise ValueError(
            "final_noise_audit_local_readout_strategy requires final_noise_audit_mitigation='readout'."
        )
    if noise_mode != "backend_scheduled" and (
        zne_scales
        or local_gate_twirling
        or dd_sequence not in {None, "", "none"}
    ):
        raise ValueError(
            "final noise audit local ZNE/gate twirling/DD currently require noise_mode='backend_scheduled'."
        )
    if noise_mode == "backend_scheduled" and zne_scales:
        _validate_backend_scheduled_local_zne_scales(
            zne_scales,
            field_name="final_noise_audit_zne_scales",
        )
    if noise_mode == "backend_scheduled":
        if runtime_profile_name != "legacy_runtime_v0":
            raise ValueError(
                "final_noise_audit_runtime_profile is only valid for final_noise_audit_mode='runtime'."
            )
        if runtime_session_policy != "prefer_session":
            raise ValueError(
                "final_noise_audit_runtime_session_policy is only valid for final_noise_audit_mode='runtime'."
            )
        if not bool(config.use_fake_backend):
            raise ValueError(
                "final noise audit backend_scheduled mode requires --final-noise-audit-use-fake-backend."
            )
        if config.backend_name in {None, ""}:
            raise ValueError(
                "final noise audit backend_scheduled mode requires --final-noise-audit-backend-name."
            )
    elif noise_mode == "runtime":
        if config.backend_name in {None, ""}:
            raise ValueError(
                "final noise audit runtime mode requires --final-noise-audit-backend-name."
            )
        if bool(config.use_fake_backend):
            raise ValueError(
                "final noise audit runtime mode requires a real runtime backend; do not enable --final-noise-audit-use-fake-backend."
            )
        if runtime_profile_name != "legacy_runtime_v0" and mitigation_mode != "none":
            raise ValueError(
                "final noise audit runtime profiles already encode mitigation/suppression; use final_noise_audit_mitigation='none' when final_noise_audit_runtime_profile is explicit."
            )
        if zne_scales or local_gate_twirling or dd_sequence not in {None, "", "none"}:
            raise ValueError(
                "final noise audit runtime full suppression stacks should use an explicit runtime profile, not local backend_scheduled knobs."
            )
    else:
        if runtime_profile_name != "legacy_runtime_v0":
            raise ValueError(
                "final_noise_audit_runtime_profile is only valid for final_noise_audit_mode='runtime'."
            )
        if runtime_session_policy != "prefer_session":
            raise ValueError(
                "final_noise_audit_runtime_session_policy is only valid for final_noise_audit_mode='runtime'."
            )


@dataclass(frozen=True)
class ResolvedAdaptStopPolicy:
    adapt_drop_floor: float
    adapt_drop_patience: int
    adapt_drop_min_depth: int
    adapt_grad_floor: float
    adapt_drop_floor_source: str
    adapt_drop_patience_source: str
    adapt_drop_min_depth_source: str
    adapt_grad_floor_source: str
    drop_policy_enabled: bool
    drop_policy_source: str
    eps_energy_termination_enabled: bool
    eps_grad_termination_enabled: bool


def _resolve_adapt_stop_policy(
    *,
    problem: str,
    continuation_mode: str,
    adapt_drop_floor: float | None,
    adapt_drop_patience: int | None,
    adapt_drop_min_depth: int | None,
    adapt_grad_floor: float | None,
) -> ResolvedAdaptStopPolicy:
    staged_problem = bool(
        str(continuation_mode).strip().lower() in _HH_STAGED_CONTINUATION_MODES
        and str(continuation_mode).strip().lower()
        in supported_continuation_modes_for_problem(str(problem).strip().lower())
    )

    def _resolve_float(raw: float | None, *, staged_value: float, default_value: float) -> tuple[float, str]:
        if raw is None:
            if staged_problem:
                return float(staged_value), "auto_staged"
            return float(default_value), "default_off"
        return float(raw), "explicit"

    def _resolve_int(raw: int | None, *, staged_value: int, default_value: int) -> tuple[int, str]:
        if raw is None:
            if staged_problem:
                return int(staged_value), "auto_staged"
            return int(default_value), "default_off"
        return int(raw), "explicit"

    drop_floor_resolved, drop_floor_source = _resolve_float(
        adapt_drop_floor,
        staged_value=-1.0,
        default_value=-1.0,
    )
    drop_patience_resolved, drop_patience_source = _resolve_int(
        adapt_drop_patience,
        staged_value=0,
        default_value=0,
    )
    drop_min_depth_resolved, drop_min_depth_source = _resolve_int(
        adapt_drop_min_depth,
        staged_value=0,
        default_value=0,
    )
    grad_floor_resolved, grad_floor_source = _resolve_float(
        adapt_grad_floor,
        staged_value=-1.0,
        default_value=-1.0,
    )
    drop_policy_enabled = bool(drop_floor_resolved >= 0.0 and drop_patience_resolved > 0)
    if staged_problem and all(src == "auto_staged" for src in (
        drop_floor_source,
        drop_patience_source,
        drop_min_depth_source,
        grad_floor_source,
    )):
        drop_policy_source = "auto_staged"
    elif any(src == "explicit" for src in (
        drop_floor_source,
        drop_patience_source,
        drop_min_depth_source,
        grad_floor_source,
    )):
        drop_policy_source = "explicit"
    else:
        drop_policy_source = "default_off"

    return ResolvedAdaptStopPolicy(
        adapt_drop_floor=float(drop_floor_resolved),
        adapt_drop_patience=int(drop_patience_resolved),
        adapt_drop_min_depth=int(drop_min_depth_resolved),
        adapt_grad_floor=float(grad_floor_resolved),
        adapt_drop_floor_source=str(drop_floor_source),
        adapt_drop_patience_source=str(drop_patience_source),
        adapt_drop_min_depth_source=str(drop_min_depth_source),
        adapt_grad_floor_source=str(grad_floor_source),
        drop_policy_enabled=bool(drop_policy_enabled),
        drop_policy_source=str(drop_policy_source),
        eps_energy_termination_enabled=(not staged_problem),
        eps_grad_termination_enabled=(not staged_problem),
    )

class _ExplicitOptionTrackingArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that records which option strings the user supplied."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        argv = list(sys.argv[1:] if args is None else args)
        parsed = super().parse_args(argv, namespace)
        explicit = {
            str(token).split("=", 1)[0]
            for token in argv
            if str(token).startswith("-")
            and str(token).split("=", 1)[0] in self._option_string_actions
        }
        setattr(parsed, "_explicit_cli_options", tuple(sorted(explicit)))
        try:
            return normalize_sr_route_profile_namespace(parsed)
        except ValueError as exc:
            self.error(str(exc))


def _build_adapt_arg_parser(*, adapt_gradient_parity_rtol: float) -> argparse.ArgumentParser:
    p = _ExplicitOptionTrackingArgumentParser(
        description=(
            "Hardcoded ADAPT-VQE static pipeline across registered problem families. "
            "Continuation defaults and staged-vs-legacy behavior resolve from the problem registry."
        )
    )
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=4.0)
    p.add_argument("--problem", choices=list(available_problem_keys()), default="hubbard")
    p.add_argument(
        "--molecular-problem-json",
        type=Path,
        default=None,
        help=(
            "JSON payload for the molecular_restricted_closed_shell pilot family. "
            "When provided, the static ADAPT front end derives the spin-orbital register size from this file."
        ),
    )
    p.add_argument(
        "--molecular-vibronic-h2-fixture-json",
        type=Path,
        default=None,
        help=(
            "molecular_vibronic_h2_fixture_v1 JSON override for the H2 vibronic family. "
            "Use this for generated projected/downfolded H2 runtime fixtures without replacing the checked fixture."
        ),
    )
    p.add_argument(
        "--molecular-vibronic-h2o-fixture-json",
        type=Path,
        default=None,
        help=(
            "Legacy molecular_vibronic_h2o_fixture_v1 JSON override for the H2O active2 one-mode "
            "smoke/prototype family. Not valid for Paper IV production H2O linear-FD evidence."
        ),
    )
    p.add_argument(
        "--molecular-vibronic-h2o-linear-fd-fixture-json",
        type=Path,
        default=None,
        help=(
            "Paper IV production all-three-mode H2O linear finite-difference fixture JSON. "
            "Required when --problem molecular_vibronic_h2o_linear_fd."
        ),
    )
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument(
        "--v-nn",
        type=float,
        default=0.0,
        help="Nearest-neighbor density interaction V for extended_hubbard and spinless_tv.",
    )
    p.add_argument(
        "--t-prime",
        type=float,
        default=0.0,
        help="Next-nearest-neighbor hopping amplitude t' for ttprime_hubbard.",
    )
    p.add_argument(
        "--n-fermions",
        type=int,
        default=None,
        help="Fixed fermion count for spinless_tv. Omit to use floor(L/2).",
    )
    p.add_argument("--omega0", type=float, default=0.0)
    p.add_argument("--g-ep", type=float, default=0.0, help="Holstein electron-phonon coupling g.")
    p.add_argument("--n-ph-max", type=int, default=1)
    p.add_argument("--boson-encoding", choices=["binary", "unary"], default="binary")
    p.add_argument("--boundary", choices=["periodic", "open"], default="open")
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    p.add_argument("--term-order", choices=["native", "sorted"], default="sorted")
    p.set_defaults(include_zero_point=True)
    p.add_argument(
        "--include-zero-point",
        dest="include_zero_point",
        action="store_true",
        help="Include the phonon zero-point identity contribution in HH Hamiltonians and exact anchors.",
    )
    p.add_argument(
        "--no-include-zero-point",
        dest="include_zero_point",
        action="store_false",
        help="Disable the phonon zero-point identity contribution in HH Hamiltonians and exact anchors.",
    )

    # ADAPT-VQE controls
    p.add_argument(
        "--adapt-pool",
        choices=list(available_adapt_pool_keys()),
        default=None,
        help=(
            "ADAPT pool family. If omitted, runtime resolves family- and continuation-aware defaults from the "
            "problem registry. full_meta means the problem-local mega pool; hamiltonian_blocks is the grouped "
            "symmetry-preserving Hamiltonian bootstrap. HH also supports opt-in scaffold-derived presets "
            "pareto_lean, pareto_lean_l3, and pareto_lean_l2."
        ),
    )
    p.add_argument(
        "--adapt-pool-class-filter-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON keep-spec for filtering the HH full_meta pool by operator class. "
            "Only valid together with --problem hh --adapt-pool full_meta."
        ),
    )
    p.add_argument(
        "--adapt-pool-label-filter-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON drop-spec for filtering the HH full_meta pool by exact label and/or prefix. "
            "Only valid together with --problem hh --adapt-pool full_meta."
        ),
    )
    p.add_argument(
        "--adapt-selected-logical-source-json",
        type=Path,
        default=None,
        help=(
            "Optional selected-logical library/artifact JSON used by the opt-in problem-generic "
            "historical selected pool filter. Ignored when --adapt-selected-logical-mode=off."
        ),
    )
    p.add_argument(
        "--adapt-selected-logical-mode",
        choices=[
            "off",
            "filter_with_full_fallback",
            "family_closure_with_full_fallback",
            "filter_fail_closed",
            "family_closure_fail_closed",
        ],
        default="off",
        help=(
            "Opt-in selected-logical pool route. off preserves the full base pool; "
            "filter_with_full_fallback keeps only matching selected historical logical generators, "
            "family_closure_with_full_fallback keeps all operators in matched historical families, "
            "and the fail_closed variants raise instead of falling back on missing/incompatible/no-match sources."
        ),
    )
    p.add_argument(
        "--adapt-selected-logical-transfer-mode",
        choices=["exact_match_v1", "boundary_v1"],
        default="exact_match_v1",
        help="Transfer rule used for selected-logical template/support-offset matches.",
    )
    p.add_argument(
        "--adapt-continuation-mode",
        choices=["legacy", "phase1_v1", "phase2_v1", "phase3_v1"],
        default=None,
        help=(
            "Continuation mode for ADAPT. If omitted, the runtime resolves the problem-aware default from the "
            "registry. phase3_v1 is the staged current path; legacy remains the explicit compatibility path."
        ),
    )
    p.add_argument(
        "--static-lane-route",
        choices=list(STATIC_LANE_ROUTE_CHOICES),
        default=STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
        help=(
            "Physical-family lane partition used by staged static ADAPT "
            "shortlisting, with problem-specific families such as HH "
            "electronic/phonon lanes, Hubbard UCCSD singles/doubles, and "
            "structured spin-boson or Bose-Hubbard full-meta lanes."
        ),
    )
    p.add_argument(
        "--physical-lane-shortlist-aggressiveness",
        type=int,
        choices=list(PHYSICAL_LANE_SHORTLIST_AGGRESSIVENESS_CHOICES),
        default=3,
        help=(
            "Physical-lane route shortlist tightening factor. Effective Phase-1 "
            "and Phase-2 shortlist caps are ceil(base/factor), and the Phase-2 "
            "shortlist fraction is base/factor."
        ),
    )
    p.add_argument(
        "--phase1-disable-lane-retention",
        dest="phase1_lane_retention_enabled",
        action="store_false",
        default=True,
        help=(
            "Paper-I diagnostic control: keep physical-lane classification "
            "and all Phase-II/III lane behavior active, but apply the Phase-I "
            "score, threshold, cap, frontier, and deterministic tie-break "
            "globally without lane quotas."
        ),
    )
    p.add_argument(
        "--sr-route-profile",
        dest="sr_route_profile_request",
        choices=list(SR_ROUTE_PROFILE_REQUEST_CHOICES),
        default=SR_ROUTE_PROFILE_REQUEST_OFF,
        help=(
            "Executable SR-SNAKE route profile. sr_snake resolves to the "
            "retained v3 conventional route with full-active-plus-singleton "
            "Phase-III response and full-ansatz supported-FS accepted refits. "
            "sr_snake_v2 preserves "
            "the 2026-07-15 window-coupled response policy and sr_snake_v1 "
            "preserves the older historical policy. "
            "sr_snake_no_novelty_metric_prune_beam_v1 preserves the v3 "
            "singleton/beam controller while bypassing Phase-II/III novelty "
            "and using metric-regularized pruning. Registered profiles fail "
            "closed on any "
            "conflicting explicit component argument; off preserves the "
            "component-level compatibility surface."
        ),
    )
    p.add_argument(
        "--historical-singleton-coordinate-solve-policy",
        choices=[
            "archival_reduced_scalar_v1",
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
        ],
        default="archival_reduced_scalar_v1",
        help=(
            "Source-locked Paper-I singleton coordinate model. The archival "
            "choice preserves the reduced scalar Schur score exactly; the "
            "supported-metric whitening v1 changes the retained singleton "
            "coordinate/Gram solve; projected-generalized v1 removes only "
            "raw-Gram null modes and solves the supported FS trust problem "
            "without whitening; global-trust v2 is reserved for an "
            "explicit SR escape profile. Neither activates the Route-A "
            "funnel or batching."
        ),
    )
    p.add_argument(
        "--historical-singleton-coordinate-solve-scope",
        choices=list(SR_COORDINATE_SOLVE_SCOPE_CHOICES),
        default=SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
        help=(
            "Phase scope for the repaired SR-SNAKE supported-metric solve. "
            "phase3_only_v1 preserves all existing SR evidence; "
            "phase2_and_phase3_v1 is the opt-in whitening-only perturbation "
            "that replaces the Phase-II scalar benefit while preserving N2, "
            "cost, lanes, singleton admission, and batching-off semantics."
        ),
    )
    p.add_argument(
        "--sr-powell-coordinate-chart-policy",
        choices=list(SR_POWELL_COORDINATE_CHART_REQUEST_CHOICES),
        default=SR_POWELL_COORDINATE_CHART_AUTO,
        help=(
            "Explicit SR-SNAKE Powell optimizer chart. auto resolves the "
            "canonical supported_whitened_adaptive_trust_v1 profile to "
            "expanded_runtime_projected_logical_v1 and all newer SR profiles "
            "to logical_shared_reduced_v1. Source-locked preferred-sequence "
            "replays must name a concrete policy."
        ),
    )
    p.add_argument(
        "--historical-singleton-trust-region-update-policy",
        choices=[
            ROUTE_A_TRUST_REGION_FIXED,
            ROUTE_A_TRUST_REGION_DISPLACEMENT_CALIBRATED_UNBOUNDED_V2,
            ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1,
        ],
        default=ROUTE_A_TRUST_REGION_FIXED,
        help=(
            "Branch-local trust-radius update on the source-locked historical "
            "singleton controller. fixed preserves the archival radius; the "
            "unbounded displacement-calibrated policies use their source-defined "
            "parameters without enabling a JR funnel; the source-metric policy "
            "uses accepted parameter displacement and requires no endpoint overlap."
        ),
    )
    p.add_argument(
        "--sr-escape-mode",
        choices=list(SR_ESCAPE_MODE_CHOICES),
        default=SR_ESCAPE_DISABLED,
        help=(
            "SR-SNAKE escape-controller depth. disabled preserves the current "
            "historical singleton route; saddle_only enables certified "
            "second-order negative-curvature escape; "
            "saddle_plus_modeled_minimum additionally enables the modeled "
            "local-minimum escape stage."
        ),
    )
    p.add_argument("--adapt-max-depth", type=int, default=20)
    p.add_argument("--adapt-eps-grad", type=float, default=1e-4)
    p.add_argument(
        "--adapt-eps-energy",
        type=float,
        default=1e-8,
        help=(
            "Energy convergence threshold. Acts as a terminating guard for legacy runs; "
            "in staged phase1_v1/phase2_v1/phase3_v1 runs it is telemetry-only."
        ),
    )
    p.add_argument(
        "--adapt-benchmark-target-abs-delta-e",
        type=float,
        default=None,
        help=(
            "Benchmark-only exact-reference stop target for offline Optuna/resource studies. "
            "When positive, ADAPT stops once |E_ADAPT - E_target| is at or below this value. "
            "E_target is the working-cutoff exact energy unless --adapt-benchmark-target-reference-energy "
            "is supplied. Do not use this as a QPU-faithful production controller input."
        ),
    )
    p.add_argument(
        "--adapt-benchmark-target-reference-energy",
        type=float,
        default=None,
        help=(
            "Optional external reference energy used only with --adapt-benchmark-target-abs-delta-e. "
            "Paper-I phonon rows use this to stop against E_ref(n_ph_ed) while retaining same-cutoff "
            "exact telemetry separately."
        ),
    )
    p.add_argument(
        "--adapt-exact-gs-override",
        type=float,
        default=None,
        help=(
            "Precomputed working-cutoff exact ground-state energy. When supplied, the pipeline records "
            "ground_state.exact_energy_source='adapt_exact_gs_override' and does not resolve the exact "
            "energy locally for this run. Intended for Optuna/CHTC batches that share one ED reference "
            "artifact across many trials."
        ),
    )
    p.add_argument(
        "--adapt-exact-gs-reference-json",
        type=Path,
        default=None,
        help=(
            "JSON manifest of precomputed working-cutoff exact energies. The pipeline matches the current "
            "problem/cutoff/Hamiltonian settings and fails closed if no entry matches. This is intended "
            "for multi-regime Optuna/CHTC batches so each subprocess reuses shared ED references."
        ),
    )
    p.add_argument(
        "--adapt-inner-optimizer",
        choices=["BFGS", "COBYLA", "POWELL", "ROTOSOLVE", "SPSA", "QNSPSA"],
        default="SPSA",
        help="Inner re-optimizer for HH seed pre-opt and per-depth ADAPT re-optimization.",
    )
    p.add_argument(
        "--adapt-state-backend",
        choices=["compiled", "legacy"],
        default="compiled",
        help="State action backend for ADAPT gradient/energy evaluations (compiled is default production path).",
    )
    p.add_argument(
        "--adapt-reopt-policy",
        choices=["append_only", "full", "windowed"],
        default="append_only",
        help=(
            "Per-depth re-optimization policy. "
            "'append_only' (default): freeze the prefix theta[:k] and optimize only the newly appended parameter. "
            "'full': legacy behavior — re-optimize all parameters jointly. "
            "'windowed': optimize a sliding window of recent parameters plus optional top-k older carry."
        ),
    )
    p.add_argument(
        "--adapt-accepted-refit-scope",
        choices=list(ACCEPTED_REFIT_SCOPE_CHOICES),
        default=ACCEPTED_REFIT_SCOPE_SELECTOR_POLICY_V1,
        help=(
            "Accepted-ansatz optimizer scope, resolved independently of the "
            "selector geometry window. selector_policy_v1 preserves existing "
            "behavior; full_ansatz_v1 refits every accepted logical parameter."
        ),
    )
    p.add_argument(
        "--adapt-accepted-refit-coordinate-chart",
        choices=list(ACCEPTED_REFIT_CHART_CHOICES),
        default=ACCEPTED_REFIT_CHART_NATIVE_V1,
        help=(
            "Accepted-ansatz optimizer coordinate chart. native_v1 preserves "
            "the current optimizer chart; supported_fs_whitened_fixed_v1 uses "
            "one fixed supported raw-Fubini-Study orthonormal chart per Powell "
            "invocation."
        ),
    )
    p.add_argument(
        "--adapt-accepted-refit-base-chart-policy",
        choices=list(ACCEPTED_REFIT_BASE_CHART_CHOICES),
        default=SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
        help=(
            "Base parameter chart whitened by the accepted-refit FS adapter. "
            "logical_shared_reduced_v1 is the primary one-coordinate-per-"
            "generator policy; expanded_runtime_projected_logical_v1 is a "
            "separately labeled redundant-chart diagnostic."
        ),
    )
    p.add_argument(
        "--adapt-window-size", type=int, default=3,
        help="Window size for 'windowed' reopt policy (number of newest parameters in active set).",
    )
    p.add_argument(
        "--adapt-window-topk", type=int, default=0,
        help="Number of older high-magnitude parameters to include in windowed active set.",
    )
    p.add_argument(
        "--phase3-geometry-window-size",
        type=int,
        default=0,
        help=(
            "Size used only by the explicit fixed_local_window_v1 Phase-III "
            "response scope; N>=1 counts the candidate coordinate. Canonical "
            "and legacy behavior are selected by --phase3-response-coordinate-scope, "
            "not by this numeric field."
        ),
    )
    p.add_argument(
        "--phase3-response-coordinate-scope",
        choices=list(PHASE3_RESPONSE_COORDINATE_SCOPE_CHOICES),
        default=PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1,
        help=(
            "Coordinates admitted to the Phase-III response model before Gram "
            "support reduction. full_active_plus_singleton_v1 is the canonical "
            "SR-SNAKE v3 policy; fixed_local_window_v1 is an explicit local "
            "ablation; legacy_reopt_coupled_v1 preserves historical v1/v2 "
            "optimizer-window coupling. This policy is independent of the "
            "Powell reoptimization/refit schedule."
        ),
    )
    p.add_argument(
        "--adapt-full-refit-every", type=int, default=0,
        help="Periodic full-prefix refit cadence for 'windowed' (0=disabled). Uses cumulative depth.",
    )
    p.add_argument(
        "--adapt-final-full-refit",
        choices=["true", "false"],
        default="true",
        help="Run a final full-prefix refit after ADAPT loop when using 'windowed' policy.",
    )
    p.add_argument(
        "--adapt-final-refit-maxiter",
        type=int,
        default=0,
        help=(
            "SPSA/SciPy maxiter for the final full-prefix refit only. "
            "0 preserves legacy behavior and reuses --adapt-maxiter."
        ),
    )
    p.add_argument(
        "--adapt-insertion-mode",
        choices=[
            "append_only",
            "adaptive",
            "full_commutation_reduced",
        ],
        default="append_only",
        help=(
            "Logical-coordinate insertion policy for staged ADAPT. "
            "'append_only' preserves existing append semantics; 'adaptive' probes insertion positions on "
            "stage-controller plateau/flat/repeated-family triggers; "
            "'full_commutation_reduced' scores one canonical representative per "
            "exactly certified commuting insertion class over the complete "
            "logical position domain every depth."
        ),
    )
    p.add_argument(
        "--phase1-energy-model",
        choices=list(PHASE1_ENERGY_MODEL_CHOICES),
        default=PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
        help="Phase-I first-order Fubini--Study trust model.",
    )
    p.add_argument("--phase1-lambda-compile", type=float, default=0.05)
    p.add_argument("--phase1-lambda-measure", type=float, default=0.02)
    p.add_argument("--phase1-lambda-leak", type=float, default=0.0)
    p.add_argument("--phase1-lambda-2q", type=float, default=CANONICAL_HARDWARE_COST_LAMBDA_2Q)
    p.add_argument("--phase1-lambda-d", type=float, default=CANONICAL_HARDWARE_COST_LAMBDA_D)
    p.add_argument("--phase1-lambda-1q", type=float, default=CANONICAL_HARDWARE_COST_LAMBDA_1Q)
    p.add_argument("--phase1-lambda-theta", type=float, default=CANONICAL_HARDWARE_COST_LAMBDA_THETA)
    p.add_argument("--phase1-lambda-shot", type=float, default=CANONICAL_HARDWARE_COST_LAMBDA_SHOT)
    p.add_argument("--phase1-score-z-alpha", type=float, default=0.0)
    p.add_argument(
        "--phase1-score-mode",
        choices=["trust_region_v1", "legacy_simple_v1"],
        default="trust_region_v1",
        help=(
            "Phase-I selector score. trust_region_v1 uses the rho-bounded "
            "Fubini-Study trust-region gain; legacy_simple_v1 preserves the "
            "older gradient-over-cost score for reproduction."
        ),
    )
    p.add_argument("--phase1-depth-ref", type=float, default=1.0)
    p.add_argument("--phase1-group-ref", type=float, default=1.0)
    p.add_argument("--phase1-shot-ref", type=float, default=1.0)
    p.add_argument("--phase1-family-ref", type=float, default=1.0)
    p.add_argument("--phase1-compile-cx-proxy-weight", type=float, default=1.0)
    p.add_argument("--phase1-compile-sq-proxy-weight", type=float, default=0.5)
    p.add_argument("--phase1-compile-rotation-step-weight", type=float, default=1.0)
    p.add_argument("--phase1-compile-position-shift-weight", type=float, default=1.0)
    p.add_argument("--phase1-compile-refit-active-weight", type=float, default=1.0)
    p.add_argument("--phase1-measure-groups-weight", type=float, default=1.0)
    p.add_argument("--phase1-measure-shots-weight", type=float, default=1.0)
    p.add_argument("--phase1-measure-reuse-weight", type=float, default=1.0)
    p.add_argument("--phase1-opt-dim-cost-scale", type=float, default=1.0)
    p.add_argument("--phase1-family-repeat-cost-scale", type=float, default=1.0)
    p.add_argument(
        "--phase1-shortlist-size",
        type=int,
        default=64,
        help="Maximum candidate count admitted into phase-1 probing before phase-2 full scoring.",
    )
    p.set_defaults(phase0_pilot_enabled=True)
    p.add_argument("--phase0-pilot-enabled", dest="phase0_pilot_enabled", action="store_true")
    p.add_argument("--phase0-no-pilot", dest="phase0_pilot_enabled", action="store_false")
    p.add_argument(
        "--phase0-pilot-alpha",
        type=float,
        default=0.1,
        help="Weak Phase-0 raw-gradient pilot alpha for DeltaE0 upper telemetry.",
    )
    p.add_argument(
        "--phase0-pilot-threshold",
        type=float,
        default=0.0,
        help="Weak Phase-0 upper-confidence DeltaE threshold; default 0 keeps all finite candidates.",
    )
    p.add_argument(
        "--phase0-pilot-max-records",
        type=int,
        default=0,
        help="Optional Phase-0 candidate-position cap; 0 means uncapped/no forced pruning.",
    )
    p.add_argument(
        "--phase0-pilot-max-operators",
        type=int,
        default=0,
        help=(
            "Optional Phase-0 macro-operator identity cap. Every insertion-position "
            "record for a retained operator remains available; 0 keeps legacy record-cap semantics."
        ),
    )
    p.add_argument("--phase1-probe-max-positions", type=int, default=6)
    p.add_argument("--phase1-plateau-patience", type=int, default=2)
    p.add_argument("--phase1-trough-margin-ratio", type=float, default=1.0)
    p.add_argument(
        "--phase2-shortlist-fraction",
        type=float,
        default=0.2,
        help="Fraction of phase-1 records admitted into phase-2 full scoring before shortlist-size capping.",
    )
    p.add_argument(
        "--phase2-shortlist-size",
        type=int,
        default=12,
        help="Maximum phase-2 shortlist size after cheap screening.",
    )
    p.add_argument(
        "--phase3-shortlist-size",
        type=int,
        default=None,
        help="Maximum unique Phase-III child identities; defaults to phase2-shortlist-size.",
    )
    p.add_argument(
        "--physical-phase2-lane-rel-threshold",
        type=float,
        default=0.10,
        help="Relative lane-health floor for physical Phase-II shortlist survival.",
    )
    p.add_argument(
        "--physical-phase1-lane-quota-pressure",
        type=float,
        default=0.70,
        help="Quota pressure for physical Phase-I lane budgets; 1 preserves all live lanes when cap allows.",
    )
    p.add_argument(
        "--physical-phase2-lane-quota-pressure",
        type=float,
        default=0.70,
        help="Quota pressure for physical Phase-II lane-health budgets; lower values favor global score rank.",
    )
    p.add_argument("--phase1-maturity-cap-min", type=int, default=None)
    p.add_argument("--phase1-maturity-cap-max", type=int, default=None)
    p.add_argument("--phase2-maturity-cap-min", type=int, default=None)
    p.add_argument("--phase2-maturity-cap-max", type=int, default=None)
    p.add_argument("--phase3-maturity-cap-min", type=int, default=None)
    p.add_argument("--phase3-maturity-cap-max", type=int, default=None)
    p.add_argument("--phase-maturity-shot-min", type=int, default=1)
    p.add_argument("--phase-maturity-shot-max", type=int, default=1)
    p.add_argument("--phase1-maturity-shot-cap", type=int, default=0)
    p.add_argument("--phase2-maturity-shot-cap", type=int, default=0)
    p.add_argument("--phase3-maturity-shot-cap", type=int, default=0)
    p.add_argument(
        "--phase2-lambda-H",
        type=float,
        default=1e-6,
        help="Phase-2 reduced-window Hessian ridge regularization lambda_H used in the inherited-window solve.",
    )
    p.add_argument(
        "--phase2-rho",
        type=float,
        default=0.25,
        help="Phase-2 trust-region radius rho used in reduced-gain scoring.",
    )
    p.add_argument(
        "--phase2-score-z-alpha",
        type=float,
        default=None,
        help="Optional Phase-2/3 confidence multiplier z_alpha. Defaults to --phase1-score-z-alpha when omitted.",
    )
    p.add_argument(
        "--phase2-curvature-policy",
        choices=list(PHASE2_CURVATURE_POLICY_CHOICES),
        default=PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
        help=(
            "Phase-II directional-curvature contract. SR-SNAKE v4 requires "
            "a finite identity-bound measured receipt and fails the run if "
            "construction is unresolved."
        ),
    )
    p.add_argument(
        "--phase2-cheap-curvature-proxy-policy",
        choices=list(PHASE2_CHEAP_CURVATURE_PROXY_POLICY_CHOICES),
        default=PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF,
        help="Phase-II cheap curvature proxies are disabled.",
    )
    p.add_argument("--phase2-depth-ref", type=float, default=1.0)
    p.add_argument("--phase2-group-ref", type=float, default=1.0)
    p.add_argument("--phase2-shot-ref", type=float, default=1.0)
    p.add_argument("--phase2-optdim-ref", type=float, default=1.0)
    p.add_argument("--phase2-reuse-ref", type=float, default=1.0)
    p.add_argument("--phase2-family-ref", type=float, default=1.0)
    p.add_argument(
        "--deferred-gram-fallback-ridge",
        type=float,
        default=1e-6,
        help=(
            "Nonnegative ridge used only when the deferred all-energy-models-"
            "infeasible Gram fallback is authorized and fires."
        ),
    )
    p.add_argument(
        "--phase2-selector-gain-mode",
        choices=["trust_region_v1", "unit_gain_v1"],
        default="trust_region_v1",
        help=(
            "Ablation-only Phase-II gain multiplier. trust_region_v1 preserves "
            "DeltaE_TR_raw; unit_gain_v1 ranks Phase-II by novelty/cost without "
            "the second-order trust-region energy factor."
        ),
    )
    p.add_argument("--phase2-cheap-score-eps", type=float, default=1e-12)
    p.add_argument("--phase2-metric-floor", type=float, default=1e-12)
    p.add_argument("--phase2-reduced-metric-collapse-rel-tol", type=float, default=1e-8)
    p.add_argument(
        "--adapt-schur-warm-start-mode",
        choices=["off", "append", "prune", "append-prune"],
        default="off",
        help=(
            "Default-off experimental Schur seed warm-starting. append enables guarded "
            "single-candidate append seeds; prune enables guarded prune compensation; "
            "append-prune enables both where geometry/mapping is supported."
        ),
    )
    p.add_argument("--phase2-ridge-growth-factor", type=float, default=10.0)
    p.add_argument("--phase2-ridge-max-steps", type=int, default=12)
    p.add_argument("--phase2-leakage-cap", type=float, default=1e6)
    p.add_argument("--phase2-compile-cx-proxy-weight", type=float, default=1.0)
    p.add_argument("--phase2-compile-sq-proxy-weight", type=float, default=0.5)
    p.add_argument("--phase2-compile-rotation-step-weight", type=float, default=1.0)
    p.add_argument("--phase2-compile-position-shift-weight", type=float, default=1.0)
    p.add_argument("--phase2-compile-refit-active-weight", type=float, default=1.0)
    p.add_argument("--phase2-measure-groups-weight", type=float, default=1.0)
    p.add_argument("--phase2-measure-shots-weight", type=float, default=1.0)
    p.add_argument("--phase2-measure-reuse-weight", type=float, default=1.0)
    p.add_argument("--phase2-opt-dim-cost-scale", type=float, default=1.0)
    p.add_argument("--phase2-family-repeat-cost-scale", type=float, default=1.0)
    p.add_argument("--phase2-lambda-2q", type=float, default=CANONICAL_HARDWARE_COST_LAMBDA_2Q)
    p.add_argument("--phase2-lambda-d", type=float, default=CANONICAL_HARDWARE_COST_LAMBDA_D)
    p.add_argument("--phase2-lambda-1q", type=float, default=CANONICAL_HARDWARE_COST_LAMBDA_1Q)
    p.add_argument("--phase2-lambda-theta", type=float, default=CANONICAL_HARDWARE_COST_LAMBDA_THETA)
    p.add_argument("--phase2-lambda-shot", type=float, default=CANONICAL_HARDWARE_COST_LAMBDA_SHOT)
    p.add_argument(
        "--phase2-w-depth",
        type=float,
        default=0.2,
        help="Phase-2 burden weight on normalized depth / compile cost in K_full.",
    )
    p.add_argument(
        "--phase2-w-group",
        type=float,
        default=0.15,
        help="Phase-2 burden weight on normalized new-group measurement cost in K_full.",
    )
    p.add_argument(
        "--phase2-w-shot",
        type=float,
        default=0.15,
        help="Phase-2 burden weight on normalized new-shot cost in K_full.",
    )
    p.add_argument(
        "--phase2-w-optdim",
        type=float,
        default=0.1,
        help="Phase-2 burden weight on normalized optimizer-dimension cost in K_full.",
    )
    p.add_argument(
        "--phase2-w-reuse",
        type=float,
        default=0.1,
        help="Phase-2 burden weight on normalized reuse penalty in K_full.",
    )
    p.add_argument(
        "--phase2-w-lifetime",
        type=float,
        default=0.05,
        help="Phase-2 burden weight on the remaining-horizon lifetime multiplier when lifetime cost mode is enabled.",
    )
    p.add_argument(
        "--phase2-eta-L",
        type=float,
        default=0.0,
        help="Phase-2 leakage penalty exponent eta_L multiplying exp(-eta_L * leakage_penalty) in the full_v2 score.",
    )
    p.add_argument(
        "--phase2-motif-bonus-weight",
        type=float,
        default=0.05,
        help="Phase-2 motif bonus weight beta_motif added on top of the reduced geometric score.",
    )
    p.add_argument(
        "--phase2-duplicate-penalty-weight",
        type=float,
        default=0.0,
        help="Phase-2 duplicate-direction penalty weight beta_dup subtracted from the augmented selector score.",
    )
    p.add_argument(
        "--phase2-frontier-ratio",
        type=float,
        default=0.9,
        help="Phase-2 shortlist frontier ratio used after cheap screening and before the full rerank.",
    )
    p.add_argument(
        "--phase3-frontier-ratio",
        type=float,
        default=0.9,
        help="Phase-3 shortlist frontier ratio used on the full-score rerank before any batch decision.",
    )
    p.add_argument(
        "--phase2-remaining-evaluations-proxy-mode",
        choices=["auto", "none", "remaining_depth"],
        default="auto",
        help=(
            "Remaining-horizon proxy mode used inside lifetime burden bookkeeping. "
            "'auto' preserves the previous behavior: remaining_depth when lifetime mode is on, otherwise none."
        ),
    )
    p.add_argument(
        "--phase3-motif-source-json",
        type=Path,
        default=None,
        help="Optional solved continuation JSON used to derive a transferable motif library for phase3_v1.",
    )
    p.add_argument(
        "--phase3-symmetry-mitigation-mode",
        choices=["off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"],
        default="off",
        help="Optional Phase 3 symmetry hook. verify_only preserves current behavior; active symmetry modes remain metadata/telemetry hooks here and are enforced in the noise oracle path.",
    )
    p.set_defaults(phase3_enable_rescue=False)
    p.add_argument(
        "--phase3-enable-rescue",
        dest="phase3_enable_rescue",
        action="store_true",
        help="Enable simulator-side phase3 rescue when an exact comparison-space state is available for the resolved problem family.",
    )
    p.add_argument(
        "--phase3-no-rescue",
        dest="phase3_enable_rescue",
        action="store_false",
        help="Disable simulator-side phase3 rescue.",
    )
    p.add_argument(
        "--phase3-lifetime-cost-mode",
        choices=["off", "phase3_v1"],
        default="phase3_v1",
        help="Enable deterministic lifetime burden weighting inside the existing full_v2 score.",
    )
    p.add_argument(
        "--phase3-hardware-cost-normalization-mode",
        choices=[
            "family_robust_v1",
            "family_robust_symmetric_arctan_v1",
            "raw_legacy_v1",
        ],
        default="family_robust_v1",
        help="Hardware-cost denominator mode. raw_legacy_v1 is source-lock compatibility only.",
    )
    p.add_argument(
        "--phase3-shadow-damping-policy",
        choices=["off", "mapped_seed_zero_query_v1"],
        default="off",
        help=(
            "Diagnostic-only damping recommendation from the existing mapped-seed "
            "receipt. It never applies damping or triggers an extra objective call."
        ),
    )
    p.add_argument(
        "--phase3-source-lock-preferred-sequence",
        type=str,
        default="",
        help="Source-lock-only JSON/list string of generator labels used to preserve archived admission ordering.",
    )
    p.add_argument(
        "--phase3-runtime-split-mode",
        type=str,
        default="off",
        help=(
            "Phase-3 runtime split mode for HH ADAPT. "
            "Canonical Route A uses shortlist_pauli_children_v1 with "
            "phase3-runtime-split-selection-mode=global_child_only_v1."
        ),
    )
    p.add_argument(
        "--allow-archival-phase3-runtime-split",
        action="store_true",
        help=(
            "Explicitly permit archival Phase-3 runtime split modes on the CLI for internal/diagnostic runs. "
            "This does not change the canonical public default, which remains split='off'."
        ),
    )
    p.add_argument(
        "--phase3-runtime-split-selection-mode",
        choices=[
            "proxy_child_set_preselection",
            "full_child_set_scoring",
            "parent_family_sum_top2_scoring",
            "archival_child_set_forward_v1",
            "global_child_only_v1",
        ],
        default="proxy_child_set_preselection",
        help=(
            "Runtime split child-set live selection mode. "
            "Only used when archival runtime split is explicitly enabled. "
            "'proxy_child_set_preselection' preserves the current main-path proxy chooser; "
            "'full_child_set_scoring' full-scores each admissible child set before choosing the live split representative; "
            "'parent_family_sum_top2_scoring' ranks the split family by the sum of its best two fully scored child sets, then instantiates the best child set from that family; "
            "'archival_child_set_forward_v1' forwards the chosen child-set representative instead of re-competing the macro parent, for reproducing archived Phase-3 split rows."
        ),
    )
    p.add_argument(
        "--phase3-runtime-split-max-subset-size",
        type=int,
        default=3,
        help=(
            "Legacy compatibility cap interpreted as cardinalities 1..N when exact subset sizes "
            "are absent. Only used when archival runtime split is explicitly enabled."
        ),
    )
    p.add_argument(
        "--phase3-runtime-split-subset-sizes",
        type=str,
        default=None,
        help="Exact allowed Pauli-word subset cardinalities, for example '2' or '1,2'.",
    )
    p.add_argument(
        "--phase3-runtime-split-child-set-symmetry-policy",
        choices=["off", "parent", "hard_guard"],
        default="parent",
        help=(
            "Symmetry policy used while enumerating archival Phase-3 runtime-split child sets. "
            "'off' applies no child-set symmetry filter. 'parent' preserves the parent candidate "
            "symmetry metadata. 'hard_guard' checks generated child sets against the fixed fermion "
            "sector. Subset cardinalities are controlled independently."
        ),
    )
    p.add_argument(
        "--phase3-runtime-split-child-padding-policy",
        choices=[
            ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1,
            ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
        ],
        default=ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1,
        help=(
            "Binary-padding policy applied to archival Route-A runtime-split "
            "children before estimator scoring. The unchecked default preserves "
            "historical compatibility; corrected Paper-I recovery runs use "
            "exact_projected_grouped_v1."
        ),
    )
    p.add_argument(
        "--adapt-child-pool-expansion-mode",
        choices=["off", "global_pauli_child_sets_v1", "pauli_child_sets_v1"],
        default="off",
        help=(
            "SNAKE global child-set pool expansion before Phase 1. "
            "'global_pauli_child_sets_v1' adds child-set candidates to the candidate pool before "
            "Phase 1/2/3 scoring; this is the SNAKE counterpart to append/Geo child-pool runs. "
            "Do not confuse this with archival Phase-3 runtime split."
        ),
    )
    p.add_argument(
        "--adapt-child-pool-expansion-symmetry-policy",
        choices=["off", "hard_guard"],
        default="off",
        help=(
            "Independent symmetry policy for global child-set pool expansion. 'hard_guard' "
            "checks the fixed fermion sector; 'off' applies no such child-set guard."
        ),
    )
    p.add_argument(
        "--adapt-child-pool-expansion-max-subset-size",
        type=int,
        default=3,
        help="Legacy compatibility cap interpreted as cardinalities 1..N when exact sizes are absent.",
    )
    p.add_argument(
        "--adapt-child-pool-expansion-subset-sizes",
        type=str,
        default=None,
        help="Exact allowed Pauli-word subset cardinalities for global child-pool expansion.",
    )
    p.add_argument(
        "--shared-pauli-pool-mode",
        choices=[
            "off",
            "shared_pauli_child_sets_v1",
            "pauli_child_sets_v1",
            "global_pauli_child_sets_v1",
            "projected_singleton_children_only_v1",
            "guarded_singleton_children_only_v1",
        ],
        default="off",
        help=(
            "Canonical Paper-I shared candidate-pool mode. "
            "'shared_pauli_child_sets_v1' builds one parent-plus-Pauli-child-set pool for SNAKE, "
            "Geo-ADAPT, and append-only ADAPT; only the downstream selection algorithm differs. "
            "'guarded_singleton_children_only_v1' removes every macro parent, globally deduplicates "
            "raw one-Pauli children, and applies fixed-sector plus legal-codeword hard guards before Phase I."
        ),
    )
    p.add_argument(
        "--shared-pauli-pool-symmetry-policy",
        choices=["off", "hard_guard"],
        default="off",
        help="Independent symmetry policy for the shared Pauli-child pool.",
    )
    p.add_argument(
        "--shared-pauli-pool-max-subset-size",
        type=int,
        default=3,
        help="Legacy compatibility cap interpreted as cardinalities 1..N when exact sizes are absent.",
    )
    p.add_argument(
        "--shared-pauli-pool-subset-sizes",
        type=str,
        default=None,
        help="Exact allowed Pauli-word subset cardinalities for the shared Pauli-child pool.",
    )
    p.add_argument(
        "--hardware-resolution-mode",
        choices=["ideal", "manual", "profile"],
        default="ideal",
        help=(
            "Gradient hardware-resolution mode for ADAPT selector scoring. "
            "'ideal' uses zero hardware/drift floors; 'manual' uses the scalar floors below; "
            "'profile' resolves a named JSON calibration profile into effective manual floors."
        ),
    )
    p.add_argument(
        "--gradient-hw-floor",
        type=float,
        default=0.0,
        help="Manual nonnegative hardware gradient floor used only with --hardware-resolution-mode manual.",
    )
    p.add_argument(
        "--gradient-drift-floor",
        type=float,
        default=0.0,
        help="Manual nonnegative drift gradient floor used only with --hardware-resolution-mode manual.",
    )
    p.add_argument(
        "--hardware-resolution-profile-json",
        type=Path,
        default=None,
        help=(
            "Explicit hardware-resolution profile manifest JSON. Required only with "
            "--hardware-resolution-mode profile."
        ),
    )
    p.add_argument(
        "--hardware-resolution-profile-name",
        type=str,
        default=None,
        help=(
            "Named gradient hardware-resolution profile within --hardware-resolution-profile-json. "
            "Required only with --hardware-resolution-mode profile."
        ),
    )
    p.add_argument(
        "--phase3-selector-policy",
        choices=["hardware_resolvable_v1", "legacy_phase3_v1", "algebraic_nested_v1"],
        default="hardware_resolvable_v1",
        help=(
            "Explicit Phase-3 selector policy/version. hardware_resolvable_v1 is the default "
            "gradient-resolution selector; legacy_phase3_v1 preserves historical controls; "
            "algebraic_nested_v1 is the preserved historical policy id and retains only its "
            "nested post-admission refit behavior; algebraic lane partitioning is retired."
        ),
    )
    p.add_argument(
        "--phase3-selector-geometry-mode",
        choices=["reduced", "proxy_reduced", "raw_exact"],
        default="reduced",
        help=(
            "Phase-3 selector geometry mode: 'reduced' is the generic canonical reduced-window selector, "
            "'proxy_reduced' uses the legacy-compatible proxy-reduced selector score on the HH phase3 surface, "
            "and 'raw_exact' ranks directly by the Phase-2 raw exact score."
        ),
    )
    p.add_argument(
        "--phase3-window-relaxation-mode",
        choices=["reduced", "no_relaxation"],
        default="reduced",
        help=(
            "Ablation-only selector switch. 'reduced' preserves canonical inherited-window "
            "Phase-3 scoring; 'no_relaxation' keeps Phase-3 active but evaluates the selector "
            "from raw candidate geometry instead of inherited-window relaxation."
        ),
    )
    p.add_argument(
        "--phase3-plateau-acquisition-mode",
        choices=list(PLATEAU_ACQUISITION_MODE_CHOICES),
        default=PLATEAU_ACQUISITION_MODE_OFF,
        help=(
            "Route-C foundation toggle. off preserves current Route-A/legacy behavior; "
            "novelty_cost_v1 identifies the plateau acquisition branch whose full ADAPT "
            "state-machine integration is intentionally deferred."
        ),
    )
    p.add_argument(
        "--phase3-plateau-unlock-margin",
        type=float,
        default=1e-8,
        help="Nonnegative Route-C unlock margin used by the future plateau commit rule.",
    )
    p.add_argument(
        "--phase3-plateau-acquisition-score",
        choices=list(PLATEAU_ACQUISITION_SCORE_CHOICES),
        default=PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
        help=(
            "Route-C plateau acquisition score. log_volume_v1 is the v1.2 QIM/log-det "
            "score; fractional_residual_v1 preserves the original N3_plat/(1+K3) selector."
        ),
    )
    p.add_argument(
        "--phase3-plateau-duplicate-policy",
        choices=list(PLATEAU_DUPLICATE_POLICY_CHOICES),
        default=PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
        help=(
            "Route-C duplicate policy. block_exact_position_v1 rejects exact candidate-position "
            "duplicates while allowing the same generator at a different insertion position."
        ),
    )
    p.add_argument(
        "--phase3-plateau-lambda-vol",
        type=float,
        default=1e-8,
        help="Positive ridge used in Route-C log_volume_v1 residual/log-det acquisition.",
    )
    p.add_argument(
        "--phase3-plateau-sigma-min",
        type=float,
        default=0.0,
        help="Nonnegative absolute residual gate for Route-C plateau acquisition.",
    )
    p.add_argument(
        "--phase3-plateau-nu-min",
        type=float,
        default=0.0,
        help="Nonnegative fractional residual gate for Route-C plateau acquisition.",
    )
    p.add_argument(
        "--phase3-plateau-volume-min",
        type=float,
        default=0.0,
        help="Nonnegative log-volume gain gate for Route-C plateau acquisition.",
    )
    p.add_argument(
        "--phase3-plateau-failed-family-patience",
        type=int,
        default=0,
        help=(
            "Route-C failed-unlock backoff by generator identity. 0 disables backoff; N blocks "
            "the same generator identity after N failed dormant admissions in the current plateau episode."
        ),
    )
    p.add_argument(
        "--phase3-plateau-seed-probe-mode",
        choices=list(PLATEAU_SEED_PROBE_MODE_CHOICES),
        default=PLATEAU_SEED_PROBE_MODE_OFF,
        help=(
            "Diagnostic Route-C plateau trial initialization. off preserves zero initialization; "
            "dormant_new_random_v1 samples finite-amplitude random seeds on dormant-plus-new "
            "coordinates before the normal inner optimizer."
        ),
    )
    p.add_argument(
        "--phase3-plateau-seed-probe-count",
        type=int,
        default=0,
        help="Number of finite-amplitude dormant/new Route-C plateau seed probes; 0 disables probes.",
    )
    p.add_argument(
        "--phase3-plateau-seed-probe-radius",
        type=float,
        default=0.0,
        help="Absolute amplitude radius for Route-C plateau seed probes; 0 disables probes.",
    )
    p.add_argument(
        "--phase3-plateau-seed-probe-seed",
        type=int,
        default=None,
        help="Optional RNG seed for Route-C plateau seed probes; omitted derives a depth-local seed from --seed.",
    )
    p.add_argument(
        "--phase3-plateau-trial-optimizer",
        choices=list(PLATEAU_TRIAL_OPTIMIZER_CHOICES),
        default=PLATEAU_TRIAL_OPTIMIZER_INHERIT,
        help=(
            "Route-C plateau trial-refit optimizer override. inherit uses the normal ADAPT inner "
            "optimizer; spsa forces SPSA for plateau trials; sp_qngd uses the state-prepared QNGD "
            "diagnostic trial optimizer only in plateau unlock attempts."
        ),
    )
    p.add_argument(
        "--phase3-plateau-trial-qngd-maxiter",
        type=int,
        default=64,
        help=(
            "Maximum SP-QNGD iterations for Route-C plateau trial refits. Used only when "
            "--phase3-plateau-trial-optimizer sp_qngd is selected."
        ),
    )
    p.add_argument(
        "--phase3-shadow-legacy-geometry-mode",
        choices=["off", "proxy_reduced", "exact_reduced"],
        default="off",
        help=(
            "Diagnostic-only HH parity hook: compute shadow legacy-style Phase-3 geometry/scoring payloads "
            "for shortlisted candidates and record them in JSON/debug output without affecting ranking or selection."
        ),
    )
    p.add_argument(
        "--phase3-shadow-legacy-max-depth",
        type=int,
        default=0,
        help="Maximum 1-based depth for --phase3-shadow-legacy-geometry-mode diagnostics (0 = all enabled depths).",
    )
    p.add_argument(
        "--phase3-backend-cost-mode",
        choices=["auto", "proxy", "transpile_single_v1", "transpile_shortlist_v1", "incremental_prefix_suffix_v1", "marrakesh_graph_span_v1"],
        default="auto",
        help=(
            "Compile-cost mode for HH Phase-3 controller scoring. "
            "auto resolves to marrakesh_graph_span_v1 for HH phase3_v1 runs; "
            "incremental_prefix_suffix_v1 uses strict prefix-aware tail transpilation; "
            "marrakesh_graph_span_v1 uses the analytic FakeMarrakesh Pauli-support graph-span estimator without transpilation; "
            "proxy remains available when explicitly requested."
        ),
    )
    p.add_argument(
        "--phase3-backend-name",
        type=str,
        default="FakeMarrakesh",
        help="Target backend name for single-backend compile-cost modes; defaults to FakeMarrakesh for the QPU-facing HH controller route.",
    )
    p.add_argument(
        "--phase3-backend-shortlist",
        type=str,
        default=None,
        help="Comma-separated backend shortlist for --phase3-backend-cost-mode transpile_shortlist_v1.",
    )
    p.add_argument(
        "--phase3-backend-transpile-seed",
        type=int,
        default=7,
        help="Seed used by the backend-conditioned transpilation oracle.",
    )
    p.add_argument(
        "--phase3-backend-optimization-level",
        type=int,
        default=1,
        help="Qiskit transpiler optimization level used by the backend-conditioned transpilation oracle.",
    )
    p.add_argument("--phase3-backend-w-2q", type=float, default=1.0, help="Weight on marginal compiled two-qubit count in the backend-aware selector cost.")
    p.add_argument("--phase3-backend-w-depth", type=float, default=0.1, help="Weight on marginal compiled serial depth in the backend-aware selector cost.")
    p.add_argument("--phase3-backend-w-size", type=float, default=0.01, help="Weight on marginal compiled circuit size in the backend-aware selector cost.")
    p.add_argument(
        "--phase3-selector-debug-topk",
        type=int,
        default=0,
        help="Emit compact phase3 selector top-k scoring logs per depth (0 disables).",
    )
    p.add_argument(
        "--phase3-selector-debug-max-depth",
        type=int,
        default=0,
        help="Maximum depth for selector debug logging (0 means all depths when enabled).",
    )
    p.add_argument(
        "--phase3-parent-collapse-debug-max-depth",
        type=int,
        default=0,
        help="Emit diagnostic-only split parent-collapse telemetry through this depth (0 disables).",
    )
    p.add_argument(
        "--phase3-oracle-gradient-mode",
        choices=list(_PHASE3_ORACLE_GRADIENT_MODE_CHOICES),
        default="off",
        help=(
            "Opt-in direct HH phase3_v1 local oracle-gradient mode. Candidate gradient scouting uses expectation or raw-shot oracle finite-difference energies; inner re-optimization stays exact unless --phase3-oracle-inner-objective-mode noisy_v1 is selected."
        ),
    )
    p.add_argument(
        "--phase3-oracle-shots",
        type=int,
        default=2048,
        help="Shots per oracle circuit when --phase3-oracle-gradient-mode is enabled.",
    )
    p.add_argument(
        "--phase3-oracle-repeats",
        type=int,
        default=1,
        help="Repeat count for oracle gradient circuits; mean aggregate only in v1.",
    )
    p.add_argument(
        "--phase3-oracle-aggregate",
        choices=["mean"],
        default="mean",
        help="Aggregate for repeated oracle gradient circuits. v1 supports mean only.",
    )
    p.add_argument(
        "--phase3-oracle-backend-name",
        type=str,
        default=None,
        help="Backend name for aer_noise/aer_density_matrix/backend_scheduled oracle modes (for example FakeNighthawk)."
    )
    p.add_argument(
        "--phase3-oracle-use-fake-backend",
        action="store_true",
        help="Use an offline fake backend for phase3 oracle-gradient mode.",
    )
    p.add_argument(
        "--phase3-oracle-seed",
        type=int,
        default=7,
        help="Seed for local oracle-gradient execution when enabled.",
    )
    p.add_argument(
        "--phase3-oracle-value-noise-model",
        choices=list(_VALUE_NOISE_MODE_CHOICES),
        default="off",
        help="Opt-in post-expectation value noise for phase3 oracle gradients; not physical shots.",
    )
    p.add_argument(
        "--phase3-oracle-value-noise-std",
        type=float,
        default=None,
        help=(
            "Gaussian iid post-expectation value-noise std in observable-value units. "
            "May be omitted when --phase3-oracle-value-noise-sigma0-abs and "
            "--phase3-oracle-value-noise-n-eff are supplied."
        ),
    )
    p.add_argument(
        "--phase3-oracle-value-noise-sigma0-abs",
        type=float,
        default=None,
        help="Shot-equivalent absolute sigma0 for phase3 oracle value noise; derives std=sigma0_abs/sqrt(N_eff).",
    )
    p.add_argument(
        "--phase3-oracle-value-noise-n-eff",
        type=float,
        default=None,
        help="Shot-equivalent effective sample count N_eff for phase3 oracle value noise.",
    )
    p.add_argument(
        "--phase3-oracle-value-noise-seed",
        type=int,
        default=None,
        help="Optional seed for phase3 oracle post-expectation value noise; defaults to oracle seed when omitted.",
    )
    p.add_argument(
        "--phase3-oracle-gradient-step",
        type=float,
        default=None,
        help="Finite-difference step for oracle-backed phase3 candidate gradients. Defaults to --adapt-finite-angle when omitted.",
    )
    p.add_argument(
        "--phase3-oracle-synthetic-depolarizing-1q-error",
        type=float,
        default=0.0,
        help="Synthetic one-qubit depolarizing probability for phase3 aer_density_matrix_synthetic_depolarizing mode.",
    )
    p.add_argument(
        "--phase3-oracle-synthetic-depolarizing-2q-error",
        type=float,
        default=0.0,
        help="Synthetic two-qubit depolarizing probability for phase3 aer_density_matrix_synthetic_depolarizing mode.",
    )
    p.add_argument(
        "--phase3-oracle-synthetic-depolarizing-1q-gates",
        type=str,
        default="x,sx,rz,h",
        help="Comma-separated one-qubit gates receiving synthetic depolarizing error.",
    )
    p.add_argument(
        "--phase3-oracle-synthetic-depolarizing-2q-gates",
        type=str,
        default="cx,cz,ecr",
        help="Comma-separated two-qubit gates receiving synthetic depolarizing error.",
    )
    p.add_argument(
        "--phase3-oracle-synthetic-coherent-1q-angle-std",
        type=float,
        default=0.0,
        help="Frozen one-qubit coherent Pauli-overrotation angle standard deviation in radians.",
    )
    p.add_argument(
        "--phase3-oracle-synthetic-coherent-2q-angle-std",
        type=float,
        default=0.0,
        help="Frozen two-qubit coherent Pauli-overrotation angle standard deviation in radians.",
    )
    p.add_argument(
        "--phase3-oracle-synthetic-coherent-seed",
        type=int,
        default=None,
        help="Optional seed for frozen gate-local coherent overrotation fields; defaults to phase3 oracle seed.",
    )
    p.add_argument(
        "--phase3-oracle-synthetic-coherent-generator-mode",
        type=str,
        default=SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT,
        help="Generator selection mode for coherent overrotation fields.",
    )
    p.add_argument(
        "--phase3-oracle-synthetic-coherent-1q-gates",
        type=str,
        default="x,sx,rx,ry,h",
        help="Comma-separated one-qubit gates receiving frozen coherent overrotations.",
    )
    p.add_argument(
        "--phase3-oracle-synthetic-coherent-2q-gates",
        type=str,
        default="cx,cz,ecr",
        help="Comma-separated two-qubit gates receiving frozen coherent overrotations.",
    )
    p.add_argument(
        "--phase3-oracle-mitigation",
        choices=["none", "readout"],
        default="none",
        help=(
            "Base mitigation mode for phase3 oracle-gradient execution. "
            "Combine with --phase3-oracle-zne-scales, --phase3-oracle-local-gate-twirling, "
            "and --phase3-oracle-dd-sequence on the backend_scheduled path."
        ),
    )
    p.add_argument(
        "--phase3-oracle-local-readout-strategy",
        choices=["mthree"],
        default=None,
        help="Local readout mitigation strategy for phase3 oracle-gradient execution.",
    )
    p.add_argument(
        "--phase3-oracle-zne-scales",
        type=str,
        default=None,
        help="Comma-separated odd integer local ZNE scales for backend_scheduled phase3 oracle-gradient execution.",
    )
    p.add_argument(
        "--phase3-oracle-local-gate-twirling",
        action="store_true",
        help="Enable local 2Q-only Pauli twirling for backend_scheduled phase3 oracle-gradient execution.",
    )
    p.add_argument(
        "--phase3-oracle-dd-sequence",
        type=str,
        default=None,
        help="Enable local DD for backend_scheduled phase3 oracle-gradient execution (currently XpXm only).",
    )
    p.add_argument(
        "--phase3-oracle-execution-surface",
        choices=["auto", "expectation_v1", "raw_measurement_v1"],
        default="auto",
        help="Execution surface for phase3 oracle-gradient mode. 'auto' selects raw-shot only for runtime with mitigation=none.",
    )
    p.add_argument(
        "--phase3-oracle-inner-objective-mode",
        choices=["exact", "noisy_v1"],
        default="exact",
        help="When noisy_v1, HH phase3_v1 inner re-optimization uses the same oracle energy surface as candidate scouting (expectation_v1 or raw_measurement_v1). The runtime path reuses parameterized compiled templates so noisy SPSA stays enabled without recompiling each evaluation from scratch.",
    )
    p.add_argument(
        "--phase3-oracle-raw-transport",
        choices=["auto", "sampler_v2"],
        default="auto",
        help="Raw transport preference when phase3 oracle execution surface resolves to raw_measurement_v1 on the runtime sampler path.",
    )
    p.add_argument(
        "--phase3-oracle-raw-store-memory",
        action="store_true",
        help="Keep emitted raw measurement records in memory during phase3 raw-shot scouting.",
    )
    p.add_argument(
        "--phase3-oracle-raw-artifact-path",
        type=str,
        default=None,
        help="Optional NDJSON(.gz) path for phase3 raw-shot measurement records.",
    )
    p.add_argument(
        "--phase3-oracle-seed-transpiler",
        type=int,
        default=None,
        help="Optional transpiler seed for phase3 oracle execution.",
    )
    p.add_argument(
        "--phase3-oracle-transpile-optimization-level",
        type=int,
        default=1,
        help="Qiskit transpiler optimization level used by phase3 oracle execution.",
    )
    p.add_argument(
        "--final-noise-audit-mode",
        choices=["off", "ideal", "shots", "aer_noise", "aer_density_matrix", "backend_scheduled", "runtime"],
        default="off",
        help=(
            "Opt-in post-run final noise audit for the canonical direct HH ADAPT path. "
            "Current support is expectation-only, including runtime and shotless Aer density-matrix audits; raw audit remains deferred."
        ),
    )
    p.add_argument(
        "--final-noise-audit-shots",
        type=int,
        default=2048,
        help="Shots per audit evaluation when --final-noise-audit-mode is enabled.",
    )
    p.add_argument(
        "--final-noise-audit-repeats",
        type=int,
        default=1,
        help="Repeat count for final noise audit evaluation; mean aggregate only in v1.",
    )
    p.add_argument(
        "--final-noise-audit-aggregate",
        choices=["mean"],
        default="mean",
        help="Aggregate for repeated final noise audit evaluations. v1 supports mean only.",
    )
    p.add_argument(
        "--final-noise-audit-backend-name",
        type=str,
        default=None,
        help="Backend name for final noise audit in aer_density_matrix, backend_scheduled, or runtime mode."
    )
    p.add_argument(
        "--final-noise-audit-use-fake-backend",
        action="store_true",
        help="Use an offline fake backend for final noise audit when supported (backend_scheduled only).",
    )
    p.add_argument(
        "--final-noise-audit-seed",
        type=int,
        default=7,
        help="Seed for final noise audit execution.",
    )
    p.add_argument(
        "--final-noise-audit-value-noise-model",
        choices=list(_VALUE_NOISE_MODE_CHOICES),
        default="off",
        help="Opt-in post-expectation value noise for the final noise audit; not physical shots.",
    )
    p.add_argument(
        "--final-noise-audit-value-noise-std",
        type=float,
        default=0.0,
        help="Gaussian iid post-expectation value-noise std in observable-value units.",
    )
    p.add_argument(
        "--final-noise-audit-value-noise-seed",
        type=int,
        default=None,
        help="Optional seed for final-audit post-expectation value noise; defaults to audit seed when omitted.",
    )
    p.add_argument(
        "--final-noise-audit-mitigation",
        choices=["none", "readout"],
        default="none",
        help=(
            "Base mitigation mode for final noise audit. Use the backend_scheduled local knobs "
            "for ZNE/twirling/DD, or a named runtime profile on the runtime path."
        ),
    )
    p.add_argument(
        "--final-noise-audit-local-readout-strategy",
        choices=["mthree"],
        default=None,
        help="Local readout mitigation strategy for backend_scheduled final noise audit when readout mitigation is enabled.",
    )
    p.add_argument(
        "--final-noise-audit-zne-scales",
        type=str,
        default=None,
        help="Comma-separated odd integer local ZNE scales for backend_scheduled final noise audit.",
    )
    p.add_argument(
        "--final-noise-audit-local-gate-twirling",
        action="store_true",
        help="Enable local 2Q-only Pauli twirling for backend_scheduled final noise audit.",
    )
    p.add_argument(
        "--final-noise-audit-dd-sequence",
        type=str,
        default=None,
        help="Enable local DD for backend_scheduled final noise audit (currently XpXm only).",
    )
    p.add_argument(
        "--final-noise-audit-runtime-profile",
        choices=[
            "legacy_runtime_v0",
            "main_twirled_readout_v1",
            "dd_probe_twirled_readout_v1",
            "final_audit_zne_twirled_readout_v1",
        ],
        default="legacy_runtime_v0",
        help="Named runtime mitigation/suppression profile for final runtime expectation audit.",
    )
    p.add_argument(
        "--final-noise-audit-runtime-session-policy",
        choices=["prefer_session", "require_session", "backend_only"],
        default="prefer_session",
        help="Runtime session policy for final runtime expectation audit.",
    )
    p.add_argument(
        "--final-noise-audit-compare-unmitigated-baseline",
        action="store_true",
        help="Also evaluate an unmitigated baseline on the same final ADAPT state for comparison.",
    )
    p.add_argument(
        "--final-noise-audit-seed-transpiler",
        type=int,
        default=None,
        help="Optional transpiler seed for final noise audit execution.",
    )
    p.add_argument(
        "--final-noise-audit-transpile-optimization-level",
        type=int,
        default=1,
        help="Qiskit transpiler optimization level used by final noise audit execution.",
    )
    p.add_argument(
        "--final-noise-audit-strict",
        action="store_true",
        help="Fail the run if final noise audit initialization or execution fails.",
    )
    p.add_argument("--adapt-maxiter", type=int, default=300, help="Inner optimizer maxiter per re-optimization")
    p.add_argument(
        "--adapt-scipy-maxfev",
        type=int,
        default=0,
        help=(
            "Optional SciPy function-evaluation cap per deterministic ADAPT re-optimization. "
            "Currently honored for Powell; 0 preserves legacy unbounded maxfev behavior."
        ),
    )
    p.add_argument("--adapt-spsa-a", type=float, default=0.2)
    p.add_argument("--adapt-spsa-c", type=float, default=0.1)
    p.add_argument("--adapt-spsa-alpha", type=float, default=0.602)
    p.add_argument("--adapt-spsa-gamma", type=float, default=0.101)
    p.add_argument("--adapt-spsa-A", type=float, default=10.0)
    p.add_argument("--adapt-spsa-avg-last", type=int, default=0)
    p.add_argument("--adapt-spsa-eval-repeats", type=int, default=1)
    p.add_argument(
        "--adapt-spsa-eval-agg",
        choices=["mean", "median"],
        default="mean",
    )
    p.add_argument("--adapt-spsa-callback-every", type=int, default=5)
    p.add_argument("--adapt-spsa-progress-every-s", type=float, default=60.0)
    p.add_argument(
        "--adapt-spsa-parallel-evaluations",
        type=int,
        default=1,
        help="Parallel objective evaluations inside one SPSA iteration; 1=serial/default.",
    )
    p.add_argument(
        "--adapt-analytic-noise-std",
        type=float,
        default=0.0,
        help="Std-dev of run-local Gaussian noise injected into exact ADAPT search-time energy and exact commutator gradients (0 = disabled).",
    )
    p.add_argument(
        "--adapt-analytic-noise-seed",
        type=int,
        default=None,
        help="Optional RNG seed for run-local ADAPT analytic Gaussian noise draws.",
    )
    p.add_argument("--adapt-seed", type=int, default=7)
    p.set_defaults(adapt_allow_repeats=True)
    p.add_argument("--adapt-allow-repeats", dest="adapt_allow_repeats", action="store_true")
    p.add_argument("--adapt-no-repeats", dest="adapt_allow_repeats", action="store_false")
    p.set_defaults(adapt_finite_angle_fallback=True)
    p.add_argument(
        "--adapt-finite-angle-fallback",
        dest="adapt_finite_angle_fallback",
        action="store_true",
        help="If gradients are below threshold, scan finite ±theta probes to continue ADAPT when beneficial.",
    )
    p.add_argument(
        "--adapt-no-finite-angle-fallback",
        dest="adapt_finite_angle_fallback",
        action="store_false",
        help="Disable finite-angle fallback and stop immediately when gradients are below threshold.",
    )
    p.add_argument(
        "--adapt-finite-angle",
        type=float,
        default=0.1,
        help="Probe angle theta used by finite-angle fallback (tests ±theta).",
    )
    p.add_argument(
        "--adapt-finite-angle-min-improvement",
        type=float,
        default=1e-12,
        help="Minimum required energy drop from finite-angle probe to accept fallback selection.",
    )
    p.add_argument(
        "--adapt-disable-hh-seed",
        action="store_true",
        help="Disable HH preconditioning with the compact quadrature seed block.",
    )
    p.add_argument(
        "--adapt-gradient-parity-check",
        action="store_true",
        help=(
            "Debug-only parity guard: compare one reused-Hpsi gradient per ADAPT depth "
            f"against the legacy commutator path (rtol={adapt_gradient_parity_rtol:.1e})."
        ),
    )
    p.add_argument(
        "--adapt-parallel-gradient-workers",
        type=int,
        default=1,
        help=(
            "Worker count for exact/noiseless ADAPT gradient-surface evaluation. "
            "Default 1 preserves serial behavior; pass 0 for CPU-aware auto sizing; values >1 use threads only for independent exact gradients."
        ),
    )
    p.add_argument(
        "--adapt-drop-floor",
        type=float,
        default=None,
        help=(
            "Energy-drop floor for plateau stop policy (drop = ΔE_abs(d-1)-ΔE_abs(d)). "
            "If omitted, staged phase1_v1/phase2_v1/phase3_v1 and legacy runs stay off. "
            "Pass a negative value to disable explicitly."
        ),
    )
    p.add_argument(
        "--adapt-drop-patience",
        type=int,
        default=None,
        help=(
            "Consecutive low-drop depth count required to trigger drop plateau stop. "
            "If omitted, staged phase1_v1/phase2_v1/phase3_v1 and legacy runs stay off."
        ),
    )
    p.add_argument(
        "--adapt-drop-min-depth",
        type=int,
        default=None,
        help=(
            "Minimum ADAPT depth before evaluating the drop plateau stop policy. "
            "If omitted, staged phase1_v1/phase2_v1/phase3_v1 and legacy runs stay off."
        ),
    )
    p.add_argument(
        "--adapt-grad-floor",
        type=float,
        default=None,
        help=(
            "Optional secondary gradient floor for drop plateau stop. "
            "If omitted, staged phase1_v1/phase2_v1/phase3_v1 and legacy runs disable it. "
            "Pass a negative value to disable explicitly."
        ),
    )
    p.add_argument(
        "--adapt-noise-floor-stop-policy",
        choices=list(_ADAPT_NOISE_FLOOR_STOP_POLICY_CHOICES),
        default="off",
        help=(
            "Opt-in conservative noise-floor stop gate. Default off preserves the legacy drop-plateau stop. "
            "noise_floor_agreement_v1 only stops when drop/plateau, runway, Phase-3 SNR, and residual-stage gates all agree."
        ),
    )
    p.add_argument(
        "--adapt-noise-floor-snr-threshold",
        type=float,
        default=2.0,
        help="Maximum current-window Phase-3 gradient SNR considered noise-floor agreement.",
    )
    p.add_argument(
        "--adapt-noise-floor-n-rem-high-threshold",
        type=float,
        default=1.0,
        help="Runway gate threshold for controller n_rem_high in noise_floor_agreement_v1.",
    )
    p.add_argument(
        "--adapt-noise-floor-useful-horizon-threshold",
        type=float,
        default=1.0,
        help="Runway gate threshold for controller useful_horizon in noise_floor_agreement_v1.",
    )
    p.add_argument(
        "--adapt-eps-energy-min-extra-depth",
        type=int,
        default=-1,
        help=(
            "Minimum extra ADAPT depth before the eps-energy guard can trigger. "
            "Use -1 to auto-set this to L. Telemetry-only in staged phase1_v1/phase2_v1/phase3_v1."
        ),
    )
    p.add_argument(
        "--adapt-eps-energy-patience",
        type=int,
        default=-1,
        help=(
            "Consecutive low-improvement depth count required for the eps-energy guard. "
            "Use -1 to auto-set this to L. Telemetry-only in staged phase1_v1/phase2_v1/phase3_v1."
        ),
    )
    p.add_argument(
        "--adapt-ref-json",
        type=Path,
        default=None,
        help=(
            "Import reference state from an ADAPT/VQE JSON initial_state.amplitudes_qn_to_q0. "
            "In HH phase1_v1/phase2_v1/phase3_v1 reruns, metadata-compatible warm/ADAPT JSON can also "
            "reuse ground_state exact-energy fields."
        ),
    )
    p.add_argument(
        "--adapt-resume-scaffold-json",
        type=Path,
        default=None,
        help=(
            "Structurally resume static HH ADAPT from a prior scaffold artifact. "
            "Mutually exclusive with --adapt-ref-json; no credential-bearing values are accepted."
        ),
    )
    p.add_argument(
        "--adapt-resume-mode",
        choices=["scaffold_v1"],
        default="scaffold_v1",
        help="Structural resume contract version. First slice supports scaffold_v1 only.",
    )
    p.add_argument(
        "--adapt-resume-boundary-refit-policy",
        choices=["required", "verified_checkpoint_no_refit_v1"],
        default="required",
        help=(
            "Boundary behavior for structural resume. required preserves the "
            "existing full-coordinate boundary refit. "
            "verified_checkpoint_no_refit_v1 may skip it only after the runtime "
            "validates a complete best-frontier checkpoint and reproduces its "
            "saved energy within the configured state-consistency tolerance."
        ),
    )
    p.add_argument(
        "--adapt-segment-id",
        type=str,
        default=None,
        help="Optional human segment identifier for static HH ADAPT boundary artifacts.",
    )
    p.add_argument(
        "--adapt-segment-target-depth",
        type=int,
        default=None,
        help="Cumulative target ansatz depth for this segment; stops before scoring once reached.",
    )
    p.add_argument(
        "--adapt-segment-target-controller-round",
        type=int,
        default=None,
        help=(
            "Optional cumulative outer-controller round target. Unlike target "
            "ansatz depth, this horizon is unaffected by accepted pruning."
        ),
    )
    p.add_argument(
        "--adapt-segment-max-new-admissions",
        type=int,
        default=None,
        help="Maximum newly admitted generators for this segment.",
    )
    p.add_argument(
        "--adapt-segment-wallclock-cap-s",
        type=float,
        default=None,
        help="Optional deterministic wall-clock safety cap in seconds for this segment.",
    )
    p.add_argument(
        "--adapt-resume-compile-smoke",
        choices=["required", "auto", "off"],
        default="auto",
        help="Local fake-backend compile smoke gate for structural resume artifacts.",
    )
    p.add_argument(
        "--adapt-resume-smoke-backend",
        type=str,
        default="FakeMarrakesh",
        help="Local fake backend name for structural resume compile smoke.",
    )
    p.add_argument("--paop-r", type=int, default=1, help="Cloud radius R for paop_full/paop_lf_full pools.")
    p.add_argument(
        "--paop-split-paulis",
        action="store_true",
        help="Split composite PAOP generators into single Pauli terms.",
    )
    p.add_argument(
        "--paop-prune-eps",
        type=float,
        default=0.0,
        help="Prune PAOP Pauli terms below this absolute coefficient threshold.",
    )
    p.add_argument(
        "--paop-normalization",
        choices=["none", "fro", "maxcoeff"],
        default="none",
        help="Normalization mode for PAOP generators before ADAPT search.",
    )

    # Trotter dynamics
    p.add_argument("--t-final", type=float, default=20.0)
    p.add_argument("--num-times", type=int, default=201)
    p.add_argument("--suzuki-order", type=int, default=2)
    p.add_argument("--trotter-steps", type=int, default=64)

    p.add_argument("--initial-state-source", choices=["exact", "adapt_vqe", "hf"], default="adapt_vqe")

    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument(
        "--adapt-current-json",
        type=Path,
        default=None,
        help=(
            "Optional partial ADAPT checkpoint JSON sidecar. When set, the runner "
            "atomically refreshes this file at safe depth boundaries so a stopped "
            "diagnostic run can recover the current scaffold/theta."
        ),
    )
    p.add_argument(
        "--adapt-current-json-every-depth",
        type=int,
        default=1,
        help="Refresh --adapt-current-json every N accepted depths; use 1 for every depth.",
    )
    p.add_argument(
        "--adapt-current-json-keep-history-tail",
        type=int,
        default=100,
        help="Maximum number of recent history rows embedded in the partial checkpoint.",
    )
    p.add_argument(
        "--adapt-estimator-call-ledger-json",
        type=Path,
        default=None,
        help=(
            "Optional full state-keyed estimator primitive ledger sidecar. When "
            "set, logical estimator accounting is instrumented and the complete "
            "ledger is written after the ADAPT run."
        ),
    )
    p.add_argument(
        "--dense-eigh-max-dim",
        type=int,
        default=8192,
        help="Skip full dense Hamiltonian diagonalization when Hilbert dimension exceeds this threshold.",
    )
    p.add_argument("--skip-pdf", action="store_true")
    p.add_argument(
        "--skip-trajectory",
        action="store_true",
        help="Skip post-ADAPT diagnostic trajectory simulation while still emitting the static JSON artifact.",
    )
    return p


@dataclass(frozen=True)
class ResolvedMainCLIConfig:
    continuation_mode: str
    phase3_oracle_gradient_config: Phase3OracleGradientConfig | None
    final_noise_audit_config: FinalNoiseAuditConfig | None


def _resolve_main_cli_configs(
    args: argparse.Namespace,
    *,
    problem: str,
    continuation_mode_resolver: Callable[..., str] = _resolve_cli_adapt_continuation_mode,
    phase3_config_cls: type[Phase3OracleGradientConfig] = Phase3OracleGradientConfig,
    phase3_resolver: Callable[[Phase3OracleGradientConfig], Phase3OracleGradientConfig] = _resolve_phase3_oracle_gradient_config,
    phase3_validator: Callable[..., None] = _validate_phase3_oracle_gradient_config,
    final_audit_config_cls: type[FinalNoiseAuditConfig] = FinalNoiseAuditConfig,
    final_audit_resolver: Callable[[FinalNoiseAuditConfig], FinalNoiseAuditConfig] = _resolve_final_noise_audit_config,
    final_audit_validator: Callable[..., None] = _validate_final_noise_audit_config,
) -> ResolvedMainCLIConfig:
    continuation_mode = continuation_mode_resolver(
        problem=str(problem),
        requested_mode=args.adapt_continuation_mode,
    )
    phase3_oracle_gradient_config: Phase3OracleGradientConfig | None = None
    final_noise_audit_config: FinalNoiseAuditConfig | None = None
    phase3_oracle_gradient_mode_key = str(args.phase3_oracle_gradient_mode).strip().lower()
    phase3_value_noise_std, phase3_value_noise_contract = _resolve_value_noise_std_contract(
        label="phase3_oracle",
        value_noise_model=str(getattr(args, "phase3_oracle_value_noise_model", "off")),
        value_noise_std=getattr(args, "phase3_oracle_value_noise_std", None),
        value_noise_sigma0_abs=getattr(args, "phase3_oracle_value_noise_sigma0_abs", None),
        value_noise_n_eff=getattr(args, "phase3_oracle_value_noise_n_eff", None),
    )
    _validate_value_noise_config(
        label="phase3_oracle",
        value_noise_model=str(getattr(args, "phase3_oracle_value_noise_model", "off")),
        value_noise_std=float(phase3_value_noise_std),
        execution_surface=(
            str(getattr(args, "phase3_oracle_execution_surface", "auto"))
            if str(getattr(args, "phase3_oracle_execution_surface", "auto")) != "auto"
            else "expectation_v1"
        ),
    )
    if (
        phase3_oracle_gradient_mode_key == "off"
        and str(getattr(args, "phase3_oracle_value_noise_model", "off")).strip().lower() != "off"
    ):
        raise ValueError("--phase3-oracle-value-noise-model requires --phase3-oracle-gradient-mode.")
    if phase3_oracle_gradient_mode_key != "off":
        phase3_oracle_gradient_config = phase3_resolver(
            phase3_config_cls(
                noise_mode=str(phase3_oracle_gradient_mode_key),
                shots=int(args.phase3_oracle_shots),
                oracle_repeats=int(args.phase3_oracle_repeats),
                oracle_aggregate=str(args.phase3_oracle_aggregate),
                backend_name=(
                    None
                    if args.phase3_oracle_backend_name in {None, ""}
                    else str(args.phase3_oracle_backend_name)
                ),
                use_fake_backend=bool(args.phase3_oracle_use_fake_backend),
                seed=int(args.phase3_oracle_seed),
                gradient_step=(
                    float(args.phase3_oracle_gradient_step)
                    if args.phase3_oracle_gradient_step is not None
                    else float(args.adapt_finite_angle)
                ),
                mitigation_mode=str(args.phase3_oracle_mitigation),
                local_readout_strategy=(
                    None
                    if args.phase3_oracle_local_readout_strategy in {None, ""}
                    else str(args.phase3_oracle_local_readout_strategy)
                ),
                zne_scales=(
                    ()
                    if args.phase3_oracle_zne_scales in {None, ""}
                    else str(args.phase3_oracle_zne_scales)
                ),
                local_gate_twirling=bool(args.phase3_oracle_local_gate_twirling),
                dd_sequence=(
                    None
                    if args.phase3_oracle_dd_sequence in {None, ""}
                    else str(args.phase3_oracle_dd_sequence)
                ),
                execution_surface_requested=str(args.phase3_oracle_execution_surface),
                raw_transport=str(args.phase3_oracle_raw_transport),
                raw_store_memory=bool(args.phase3_oracle_raw_store_memory),
                raw_artifact_path=(
                    None
                    if args.phase3_oracle_raw_artifact_path in {None, ""}
                    else str(args.phase3_oracle_raw_artifact_path)
                ),
                seed_transpiler=(
                    None
                    if args.phase3_oracle_seed_transpiler is None
                    else int(args.phase3_oracle_seed_transpiler)
                ),
                transpile_optimization_level=int(args.phase3_oracle_transpile_optimization_level),
                value_noise_model=str(args.phase3_oracle_value_noise_model),
                value_noise_std=float(phase3_value_noise_std),
                value_noise_seed=(
                    None
                    if args.phase3_oracle_value_noise_seed is None
                    else int(args.phase3_oracle_value_noise_seed)
                ),
                value_noise_sigma0_abs=phase3_value_noise_contract["sigma0_abs"],
                value_noise_n_eff=phase3_value_noise_contract["N_eff"],
                value_noise_semantic=str(phase3_value_noise_contract["semantic"]),
                value_noise_std_source=str(phase3_value_noise_contract["std_source"]),
                synthetic_depolarizing_1q_error=float(args.phase3_oracle_synthetic_depolarizing_1q_error),
                synthetic_depolarizing_2q_error=float(args.phase3_oracle_synthetic_depolarizing_2q_error),
                synthetic_depolarizing_1q_gates=_parse_gate_name_tuple(
                    args.phase3_oracle_synthetic_depolarizing_1q_gates,
                    default=SYNTHETIC_DEPOLARIZING_1Q_GATES_DEFAULT,
                    field_name="phase3_oracle_synthetic_depolarizing_1q_gates",
                ),
                synthetic_depolarizing_2q_gates=_parse_gate_name_tuple(
                    args.phase3_oracle_synthetic_depolarizing_2q_gates,
                    default=SYNTHETIC_DEPOLARIZING_2Q_GATES_DEFAULT,
                    field_name="phase3_oracle_synthetic_depolarizing_2q_gates",
                ),
                synthetic_coherent_1q_angle_std=float(args.phase3_oracle_synthetic_coherent_1q_angle_std),
                synthetic_coherent_2q_angle_std=float(args.phase3_oracle_synthetic_coherent_2q_angle_std),
                synthetic_coherent_seed=(
                    None
                    if args.phase3_oracle_synthetic_coherent_seed is None
                    else int(args.phase3_oracle_synthetic_coherent_seed)
                ),
                synthetic_coherent_generator_mode=str(
                    args.phase3_oracle_synthetic_coherent_generator_mode
                ),
                synthetic_coherent_1q_gates=_parse_gate_name_tuple(
                    args.phase3_oracle_synthetic_coherent_1q_gates,
                    default=SYNTHETIC_COHERENT_1Q_GATES_DEFAULT,
                    field_name="phase3_oracle_synthetic_coherent_1q_gates",
                ),
                synthetic_coherent_2q_gates=_parse_gate_name_tuple(
                    args.phase3_oracle_synthetic_coherent_2q_gates,
                    default=SYNTHETIC_COHERENT_2Q_GATES_DEFAULT,
                    field_name="phase3_oracle_synthetic_coherent_2q_gates",
                ),
            )
        )
        phase3_validator(
            config=phase3_oracle_gradient_config,
            problem=str(problem),
            continuation_mode=str(continuation_mode),
        )
    final_noise_audit_mode_key = str(args.final_noise_audit_mode).strip().lower()
    _validate_value_noise_config(
        label="final_noise_audit",
        value_noise_model=str(getattr(args, "final_noise_audit_value_noise_model", "off")),
        value_noise_std=float(getattr(args, "final_noise_audit_value_noise_std", 0.0)),
        execution_surface="expectation_v1",
    )
    if (
        final_noise_audit_mode_key == "off"
        and str(getattr(args, "final_noise_audit_value_noise_model", "off")).strip().lower() != "off"
    ):
        raise ValueError("--final-noise-audit-value-noise-model requires --final-noise-audit-mode.")
    if final_noise_audit_mode_key != "off":
        final_noise_audit_config = final_audit_resolver(
            final_audit_config_cls(
                noise_mode=str(final_noise_audit_mode_key),
                shots=int(args.final_noise_audit_shots),
                oracle_repeats=int(args.final_noise_audit_repeats),
                oracle_aggregate=str(args.final_noise_audit_aggregate),
                backend_name=(
                    None
                    if args.final_noise_audit_backend_name in {None, ""}
                    else str(args.final_noise_audit_backend_name)
                ),
                use_fake_backend=bool(args.final_noise_audit_use_fake_backend),
                seed=int(args.final_noise_audit_seed),
                mitigation_mode=str(args.final_noise_audit_mitigation),
                local_readout_strategy=(
                    None
                    if args.final_noise_audit_local_readout_strategy in {None, ""}
                    else str(args.final_noise_audit_local_readout_strategy)
                ),
                zne_scales=(
                    ()
                    if args.final_noise_audit_zne_scales in {None, ""}
                    else str(args.final_noise_audit_zne_scales)
                ),
                local_gate_twirling=bool(args.final_noise_audit_local_gate_twirling),
                dd_sequence=(
                    None
                    if args.final_noise_audit_dd_sequence in {None, ""}
                    else str(args.final_noise_audit_dd_sequence)
                ),
                runtime_profile_name=str(args.final_noise_audit_runtime_profile),
                runtime_session_policy=str(args.final_noise_audit_runtime_session_policy),
                compare_unmitigated_baseline=bool(
                    args.final_noise_audit_compare_unmitigated_baseline
                ),
                seed_transpiler=(
                    None
                    if args.final_noise_audit_seed_transpiler is None
                    else int(args.final_noise_audit_seed_transpiler)
                ),
                transpile_optimization_level=int(
                    args.final_noise_audit_transpile_optimization_level
                ),
                strict=bool(args.final_noise_audit_strict),
                value_noise_model=str(args.final_noise_audit_value_noise_model),
                value_noise_std=float(args.final_noise_audit_value_noise_std),
                value_noise_seed=(
                    None
                    if args.final_noise_audit_value_noise_seed is None
                    else int(args.final_noise_audit_value_noise_seed)
                ),
            )
        )
        final_audit_validator(
            config=final_noise_audit_config,
            problem=str(problem),
        )
    return ResolvedMainCLIConfig(
        continuation_mode=str(continuation_mode),
        phase3_oracle_gradient_config=phase3_oracle_gradient_config,
        final_noise_audit_config=final_noise_audit_config,
    )

def _build_run_hardcoded_adapt_vqe_kwargs(
    args: argparse.Namespace,
    *,
    h_poly: Any,
    resolved_problem_context: Any | None = None,
    cli_adapt_continuation_mode: str,
    adapt_ref_base_depth: int,
    psi_ref_override: Any,
    psi_ref_source: str | None,
    psi_ref_handoff_state_kind: str | None,
    exact_gs_override: float,
    phase3_oracle_gradient_config: Phase3OracleGradientConfig | None,
    final_noise_audit_config: FinalNoiseAuditConfig | None,
    route_profile_already_normalized: bool = False,
) -> dict[str, Any]:
    # Direct unit/integration callers can bypass the top-level ``parse_args``
    # helper, so repeat the idempotent route-profile normalization here before
    # any batching aliases or runtime kwargs are derived.  A prefactored deep
    # runner may explicitly prove that it received this Namespace from the
    # normalizing parser and skip only this duplicate defensive pass.
    if not route_profile_already_normalized:
        normalize_sr_route_profile_namespace(args)

    child_padding_policy = str(
        getattr(
            args,
            "phase3_runtime_split_child_padding_policy",
            ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1,
        )
    )
    if child_padding_policy == ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1:
        route_a_child_padding_config = RouteAChildPaddingConfig(
            policy=child_padding_policy
        )
    else:
        if resolved_problem_context is None:
            raise ValueError(
                "Active Route-A child-padding enforcement requires a resolved "
                "problem context."
            )
        route_a_child_padding_config = RouteAChildPaddingConfig(
            policy=child_padding_policy,
            problem_key=str(args.problem),
            num_sites=int(args.L),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            total_register_width=int(resolved_problem_context.layout.total_qubits),
        )
    return {
        "h_poly": h_poly,
        "resolved_problem_context": resolved_problem_context,
        "num_sites": int(args.L),
        "ordering": str(args.ordering),
        "problem": str(args.problem),
        "molecular_problem_json": (
            None
            if args.molecular_problem_json is None
            else str(Path(args.molecular_problem_json))
        ),
        "molecular_vibronic_h2_fixture_json": (
            None
            if args.molecular_vibronic_h2_fixture_json is None
            else str(Path(args.molecular_vibronic_h2_fixture_json))
        ),
        "molecular_vibronic_h2o_fixture_json": (
            None
            if args.molecular_vibronic_h2o_fixture_json is None
            else str(Path(args.molecular_vibronic_h2o_fixture_json))
        ),
        "molecular_vibronic_h2o_linear_fd_fixture_json": (
            None
            if args.molecular_vibronic_h2o_linear_fd_fixture_json is None
            else str(Path(args.molecular_vibronic_h2o_linear_fd_fixture_json))
        ),
        "adapt_pool": str(args.adapt_pool) if args.adapt_pool is not None else None,
        # Historical route registries are non-authorizing observations only.
        # The legacy CLI no longer accepts route/profile selection controls.
        "static_route_id": "unspecified",
        "static_meta_feature_profile": "off",
        "static_lane_route": str(
            getattr(
                args,
                "static_lane_route",
                STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
            )
        ),
        "physical_lane_shortlist_aggressiveness": int(
            getattr(args, "physical_lane_shortlist_aggressiveness", 3)
        ),
        "phase1_lane_retention_enabled": bool(
            getattr(args, "phase1_lane_retention_enabled", True)
        ),
        "sr_route_profile_request": str(
            getattr(args, "sr_route_profile_request", SR_ROUTE_PROFILE_REQUEST_OFF)
        ),
        "sr_route_profile_resolved": getattr(
            args, "sr_route_profile_resolved", None
        ),
        "sr_route_profile_contract": getattr(
            args, "sr_route_profile_contract", None
        ),
        "sr_route_profile_contract_sha256": getattr(
            args, "sr_route_profile_contract_sha256", None
        ),
        "historical_singleton_coordinate_solve_policy": str(
            getattr(
                args,
                "historical_singleton_coordinate_solve_policy",
                "archival_reduced_scalar_v1",
            )
        ),
        "historical_singleton_coordinate_solve_scope": str(
            getattr(
                args,
                "historical_singleton_coordinate_solve_scope",
                SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
            )
        ),
        "phase2_gram_novelty_policy": str(
            getattr(
                args,
                "phase2_gram_novelty_policy",
                "off",
            )
        ),
        "phase3_gram_novelty_policy": str(
            getattr(
                args,
                "phase3_gram_novelty_policy",
                "off",
            )
        ),
        "sr_controller_ablation_contract": str(
            getattr(
                args,
                "sr_controller_ablation_contract",
                SR_CONTROLLER_ABLATION_CONTRACT_OFF,
            )
        ),
        "sr_powell_coordinate_chart_policy": str(
            getattr(
                args,
                "sr_powell_coordinate_chart_policy",
                SR_POWELL_COORDINATE_CHART_AUTO,
            )
        ),
        "historical_singleton_trust_region_update_policy": str(
            getattr(
                args,
                "historical_singleton_trust_region_update_policy",
                ROUTE_A_TRUST_REGION_FIXED,
            )
        ),
        "sr_escape_mode": str(
            getattr(args, "sr_escape_mode", SR_ESCAPE_DISABLED)
        ),
        "t": float(args.t),
        "u": float(args.u),
        "dv": float(args.dv),
        "v_nn": float(args.v_nn),
        "t_prime": float(args.t_prime),
        "n_fermions": (
            None if args.n_fermions is None else int(args.n_fermions)
        ),
        "boundary": str(args.boundary),
        "omega0": float(args.omega0),
        "g_ep": float(args.g_ep),
        "n_ph_max": int(args.n_ph_max),
        "boson_encoding": str(args.boson_encoding),
        "include_zero_point": bool(args.include_zero_point),
        "max_depth": int(args.adapt_max_depth),
        "eps_grad": float(args.adapt_eps_grad),
        "eps_energy": float(args.adapt_eps_energy),
        "benchmark_target_abs_delta_e": (
            None
            if args.adapt_benchmark_target_abs_delta_e is None
            else float(args.adapt_benchmark_target_abs_delta_e)
        ),
        "benchmark_target_reference_energy": (
            None
            if args.adapt_benchmark_target_reference_energy is None
            else float(args.adapt_benchmark_target_reference_energy)
        ),
        "adapt_current_json": (
            Path(args.adapt_current_json)
            if getattr(args, "adapt_current_json", None) is not None
            else None
        ),
        "adapt_current_json_every_depth": int(args.adapt_current_json_every_depth),
        "adapt_current_json_keep_history_tail": int(args.adapt_current_json_keep_history_tail),
        "adapt_estimator_call_ledger_enabled": bool(
            getattr(args, "adapt_estimator_call_ledger_json", None) is not None
        ),
        "maxiter": int(args.adapt_maxiter),
        "adapt_scipy_maxfev": int(args.adapt_scipy_maxfev),
        "seed": int(args.adapt_seed),
        "adapt_inner_optimizer": str(args.adapt_inner_optimizer),
        "adapt_spsa_a": float(args.adapt_spsa_a),
        "adapt_spsa_c": float(args.adapt_spsa_c),
        "adapt_spsa_alpha": float(args.adapt_spsa_alpha),
        "adapt_spsa_gamma": float(args.adapt_spsa_gamma),
        "adapt_spsa_A": float(args.adapt_spsa_A),
        "adapt_spsa_avg_last": int(args.adapt_spsa_avg_last),
        "adapt_spsa_eval_repeats": int(args.adapt_spsa_eval_repeats),
        "adapt_spsa_eval_agg": str(args.adapt_spsa_eval_agg),
        "adapt_spsa_callback_every": int(args.adapt_spsa_callback_every),
        "adapt_spsa_progress_every_s": float(args.adapt_spsa_progress_every_s),
        "adapt_spsa_parallel_evaluations": int(args.adapt_spsa_parallel_evaluations),
        "adapt_analytic_noise_std": float(args.adapt_analytic_noise_std),
        "adapt_analytic_noise_seed": None
                if args.adapt_analytic_noise_seed is None
                else int(args.adapt_analytic_noise_seed),
        "adapt_state_backend": str(args.adapt_state_backend),
        "adapt_reopt_policy": str(args.adapt_reopt_policy),
        "adapt_accepted_refit_scope": str(args.adapt_accepted_refit_scope),
        "adapt_accepted_refit_coordinate_chart": str(
            args.adapt_accepted_refit_coordinate_chart
        ),
        "adapt_accepted_refit_base_chart_policy": str(
            args.adapt_accepted_refit_base_chart_policy
        ),
        "adapt_window_size": int(args.adapt_window_size),
        "adapt_window_topk": int(args.adapt_window_topk),
        "phase3_geometry_window_size": int(args.phase3_geometry_window_size),
        "phase3_response_coordinate_scope": str(
            args.phase3_response_coordinate_scope
        ),
        "adapt_full_refit_every": int(args.adapt_full_refit_every),
        "adapt_final_full_refit": bool(str(args.adapt_final_full_refit).strip().lower() == "true"),
        "adapt_final_refit_maxiter": int(args.adapt_final_refit_maxiter),
        "adapt_insertion_mode": str(args.adapt_insertion_mode),
        "adapt_continuation_mode": str(cli_adapt_continuation_mode),
        "allow_repeats": bool(args.adapt_allow_repeats),
        "finite_angle_fallback": bool(args.adapt_finite_angle_fallback),
        "finite_angle": float(args.adapt_finite_angle),
        "finite_angle_min_improvement": float(args.adapt_finite_angle_min_improvement),
        "adapt_drop_floor": float(args.adapt_drop_floor) if args.adapt_drop_floor is not None else None,
        "adapt_drop_patience": int(args.adapt_drop_patience) if args.adapt_drop_patience is not None else None,
        "adapt_drop_min_depth": int(args.adapt_drop_min_depth) if args.adapt_drop_min_depth is not None else None,
        "adapt_grad_floor": float(args.adapt_grad_floor) if args.adapt_grad_floor is not None else None,
        "adapt_noise_floor_stop_policy": str(args.adapt_noise_floor_stop_policy),
        "adapt_noise_floor_snr_threshold": float(args.adapt_noise_floor_snr_threshold),
        "adapt_noise_floor_n_rem_high_threshold": float(args.adapt_noise_floor_n_rem_high_threshold),
        "adapt_noise_floor_useful_horizon_threshold": float(args.adapt_noise_floor_useful_horizon_threshold),
        "adapt_eps_energy_min_extra_depth": int(args.adapt_eps_energy_min_extra_depth),
        "adapt_eps_energy_patience": int(args.adapt_eps_energy_patience),
        "adapt_ref_base_depth": int(adapt_ref_base_depth),
        "paop_r": int(args.paop_r),
        "paop_split_paulis": bool(args.paop_split_paulis),
        "paop_prune_eps": float(args.paop_prune_eps),
        "paop_normalization": str(args.paop_normalization),
        "disable_hh_seed": bool(args.adapt_disable_hh_seed),
        "psi_ref_override": psi_ref_override,
        "psi_ref_source": psi_ref_source,
        "psi_ref_handoff_state_kind": psi_ref_handoff_state_kind,
        "adapt_resume_scaffold_json": (
            Path(args.adapt_resume_scaffold_json)
            if getattr(args, "adapt_resume_scaffold_json", None) is not None
            else None
        ),
        "adapt_resume_mode": str(getattr(args, "adapt_resume_mode", "scaffold_v1")),
        "adapt_resume_boundary_refit_policy": str(
            getattr(args, "adapt_resume_boundary_refit_policy", "required")
        ),
        "adapt_segment_id": (
            None
            if getattr(args, "adapt_segment_id", None) in {None, ""}
            else str(args.adapt_segment_id)
        ),
        "adapt_segment_target_depth": (
            None
            if getattr(args, "adapt_segment_target_depth", None) is None
            else int(args.adapt_segment_target_depth)
        ),
        "adapt_segment_target_controller_round": (
            None
            if getattr(args, "adapt_segment_target_controller_round", None) is None
            else int(args.adapt_segment_target_controller_round)
        ),
        "adapt_segment_max_new_admissions": (
            None
            if getattr(args, "adapt_segment_max_new_admissions", None) is None
            else int(args.adapt_segment_max_new_admissions)
        ),
        "adapt_segment_wallclock_cap_s": (
            None
            if getattr(args, "adapt_segment_wallclock_cap_s", None) is None
            else float(args.adapt_segment_wallclock_cap_s)
        ),
        "adapt_resume_compile_smoke": str(getattr(args, "adapt_resume_compile_smoke", "auto")),
        "adapt_resume_smoke_backend": str(getattr(args, "adapt_resume_smoke_backend", "FakeMarrakesh")),
        "adapt_gradient_parity_check": bool(args.adapt_gradient_parity_check),
        "adapt_parallel_gradient_workers": int(args.adapt_parallel_gradient_workers),
        "exact_gs_override": float(exact_gs_override),
        "phase1_energy_model": str(
            getattr(
                args,
                "phase1_energy_model",
                PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
            )
        ),
        "phase1_lambda_compile": float(args.phase1_lambda_compile),
        "phase1_lambda_measure": float(args.phase1_lambda_measure),
        "phase1_lambda_leak": float(args.phase1_lambda_leak),
        "phase1_lambda_2q": None if args.phase1_lambda_2q is None else float(args.phase1_lambda_2q),
        "phase1_lambda_d": None if args.phase1_lambda_d is None else float(args.phase1_lambda_d),
        "phase1_lambda_1q": None if args.phase1_lambda_1q is None else float(args.phase1_lambda_1q),
        "phase1_lambda_theta": None if args.phase1_lambda_theta is None else float(args.phase1_lambda_theta),
        "phase1_lambda_shot": None if args.phase1_lambda_shot is None else float(args.phase1_lambda_shot),
        "phase1_score_z_alpha": float(args.phase1_score_z_alpha),
        "phase1_score_mode": str(args.phase1_score_mode),
        "phase1_depth_ref": float(args.phase1_depth_ref),
        "phase1_group_ref": float(args.phase1_group_ref),
        "phase1_shot_ref": float(args.phase1_shot_ref),
        "phase1_family_ref": float(args.phase1_family_ref),
        "phase1_compile_cx_proxy_weight": float(args.phase1_compile_cx_proxy_weight),
        "phase1_compile_sq_proxy_weight": float(args.phase1_compile_sq_proxy_weight),
        "phase1_compile_rotation_step_weight": float(args.phase1_compile_rotation_step_weight),
        "phase1_compile_position_shift_weight": float(args.phase1_compile_position_shift_weight),
        "phase1_compile_refit_active_weight": float(args.phase1_compile_refit_active_weight),
        "phase1_measure_groups_weight": float(args.phase1_measure_groups_weight),
        "phase1_measure_shots_weight": float(args.phase1_measure_shots_weight),
        "phase1_measure_reuse_weight": float(args.phase1_measure_reuse_weight),
        "phase1_opt_dim_cost_scale": float(args.phase1_opt_dim_cost_scale),
        "phase1_family_repeat_cost_scale": float(args.phase1_family_repeat_cost_scale),
        "phase1_shortlist_size": int(args.phase1_shortlist_size),
        "phase0_pilot_enabled": bool(args.phase0_pilot_enabled),
        "phase0_pilot_alpha": float(args.phase0_pilot_alpha),
        "phase0_pilot_threshold": float(args.phase0_pilot_threshold),
        "phase0_pilot_max_records": int(args.phase0_pilot_max_records),
        "phase0_pilot_max_operators": int(args.phase0_pilot_max_operators),
        "phase1_probe_max_positions": int(args.phase1_probe_max_positions),
        "phase1_plateau_patience": int(args.phase1_plateau_patience),
        "phase1_trough_margin_ratio": float(args.phase1_trough_margin_ratio),
        "phase2_shortlist_fraction": float(args.phase2_shortlist_fraction),
        "phase2_shortlist_size": int(args.phase2_shortlist_size),
        "phase3_shortlist_size": (
            None
            if args.phase3_shortlist_size is None
            else int(args.phase3_shortlist_size)
        ),
        "physical_phase2_lane_rel_threshold": float(args.physical_phase2_lane_rel_threshold),
        "physical_phase1_lane_quota_pressure": float(args.physical_phase1_lane_quota_pressure),
        "physical_phase2_lane_quota_pressure": float(args.physical_phase2_lane_quota_pressure),
        "phase1_maturity_cap_min": None if args.phase1_maturity_cap_min is None else int(args.phase1_maturity_cap_min),
        "phase1_maturity_cap_max": None if args.phase1_maturity_cap_max is None else int(args.phase1_maturity_cap_max),
        "phase2_maturity_cap_min": None if args.phase2_maturity_cap_min is None else int(args.phase2_maturity_cap_min),
        "phase2_maturity_cap_max": None if args.phase2_maturity_cap_max is None else int(args.phase2_maturity_cap_max),
        "phase3_maturity_cap_min": None if args.phase3_maturity_cap_min is None else int(args.phase3_maturity_cap_min),
        "phase3_maturity_cap_max": None if args.phase3_maturity_cap_max is None else int(args.phase3_maturity_cap_max),
        "phase_maturity_shot_min": int(args.phase_maturity_shot_min),
        "phase_maturity_shot_max": int(args.phase_maturity_shot_max),
        "phase1_maturity_shot_cap": int(args.phase1_maturity_shot_cap),
        "phase2_maturity_shot_cap": int(args.phase2_maturity_shot_cap),
        "phase3_maturity_shot_cap": int(args.phase3_maturity_shot_cap),
        "phase2_lambda_H": float(args.phase2_lambda_H),
        "phase2_rho": float(args.phase2_rho),
        "phase2_score_z_alpha": float(args.phase2_score_z_alpha)
                if args.phase2_score_z_alpha is not None
                else None,
        "phase2_curvature_policy": str(
            getattr(
                args,
                "phase2_curvature_policy",
                PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
            )
        ),
        "phase2_cheap_curvature_proxy_policy": str(
            getattr(
                args,
                "phase2_cheap_curvature_proxy_policy",
                PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF,
            )
        ),
        "phase2_depth_ref": float(args.phase2_depth_ref),
        "phase2_group_ref": float(args.phase2_group_ref),
        "phase2_shot_ref": float(args.phase2_shot_ref),
        "phase2_optdim_ref": float(args.phase2_optdim_ref),
        "phase2_reuse_ref": float(args.phase2_reuse_ref),
        "phase2_family_ref": float(args.phase2_family_ref),
        "deferred_gram_fallback_ridge": float(
            args.deferred_gram_fallback_ridge
        ),
        "phase2_selector_gain_mode": str(args.phase2_selector_gain_mode),
        "phase2_cheap_score_eps": float(args.phase2_cheap_score_eps),
        "phase2_metric_floor": float(args.phase2_metric_floor),
        "phase2_reduced_metric_collapse_rel_tol": float(
                args.phase2_reduced_metric_collapse_rel_tol
            ),
        "adapt_schur_warm_start_mode": str(getattr(args, "adapt_schur_warm_start_mode", "off")),
        "phase2_ridge_growth_factor": float(args.phase2_ridge_growth_factor),
        "phase2_ridge_max_steps": int(args.phase2_ridge_max_steps),
        "phase2_leakage_cap": float(args.phase2_leakage_cap),
        "phase2_compile_cx_proxy_weight": float(args.phase2_compile_cx_proxy_weight),
        "phase2_compile_sq_proxy_weight": float(args.phase2_compile_sq_proxy_weight),
        "phase2_compile_rotation_step_weight": float(args.phase2_compile_rotation_step_weight),
        "phase2_compile_position_shift_weight": float(args.phase2_compile_position_shift_weight),
        "phase2_compile_refit_active_weight": float(args.phase2_compile_refit_active_weight),
        "phase2_measure_groups_weight": float(args.phase2_measure_groups_weight),
        "phase2_measure_shots_weight": float(args.phase2_measure_shots_weight),
        "phase2_measure_reuse_weight": float(args.phase2_measure_reuse_weight),
        "phase2_opt_dim_cost_scale": float(args.phase2_opt_dim_cost_scale),
        "phase2_family_repeat_cost_scale": float(args.phase2_family_repeat_cost_scale),
        "phase2_lambda_2q": None if args.phase2_lambda_2q is None else float(args.phase2_lambda_2q),
        "phase2_lambda_d": None if args.phase2_lambda_d is None else float(args.phase2_lambda_d),
        "phase2_lambda_1q": None if args.phase2_lambda_1q is None else float(args.phase2_lambda_1q),
        "phase2_lambda_theta": None if args.phase2_lambda_theta is None else float(args.phase2_lambda_theta),
        "phase2_lambda_shot": None if args.phase2_lambda_shot is None else float(args.phase2_lambda_shot),
        "phase2_w_depth": float(args.phase2_w_depth),
        "phase2_w_group": float(args.phase2_w_group),
        "phase2_w_shot": float(args.phase2_w_shot),
        "phase2_w_optdim": float(args.phase2_w_optdim),
        "phase2_w_reuse": float(args.phase2_w_reuse),
        "phase2_w_lifetime": float(args.phase2_w_lifetime),
        "phase2_eta_L": float(args.phase2_eta_L),
        "phase2_motif_bonus_weight": float(args.phase2_motif_bonus_weight),
        "phase2_duplicate_penalty_weight": float(args.phase2_duplicate_penalty_weight),
        "phase2_frontier_ratio": float(args.phase2_frontier_ratio),
        "phase3_frontier_ratio": float(args.phase3_frontier_ratio),
        "phase2_remaining_evaluations_proxy_mode": str(
                args.phase2_remaining_evaluations_proxy_mode
            ),
        "adapt_pool_class_filter_json": Path(args.adapt_pool_class_filter_json)
                if args.adapt_pool_class_filter_json is not None
                else None,
        "adapt_pool_label_filter_json": Path(args.adapt_pool_label_filter_json)
                if args.adapt_pool_label_filter_json is not None
                else None,
        "adapt_selected_logical_source_json": Path(args.adapt_selected_logical_source_json)
                if args.adapt_selected_logical_source_json is not None
                else None,
        "adapt_selected_logical_mode": str(args.adapt_selected_logical_mode),
        "adapt_selected_logical_transfer_mode": str(args.adapt_selected_logical_transfer_mode),
        "phase3_motif_source_json": Path(args.phase3_motif_source_json) if args.phase3_motif_source_json is not None else None,
        "phase3_symmetry_mitigation_mode": str(args.phase3_symmetry_mitigation_mode),
        "phase3_enable_rescue": bool(args.phase3_enable_rescue),
        "phase3_lifetime_cost_mode": str(args.phase3_lifetime_cost_mode),
        "phase3_hardware_cost_normalization_mode": str(args.phase3_hardware_cost_normalization_mode),
        "phase3_shadow_damping_policy": str(args.phase3_shadow_damping_policy),
        "phase3_source_lock_preferred_sequence": str(args.phase3_source_lock_preferred_sequence),
        "phase3_runtime_split_mode": str(args.phase3_runtime_split_mode),
        "phase3_runtime_split_selection_mode": str(args.phase3_runtime_split_selection_mode),
        "phase3_runtime_split_max_subset_size": int(args.phase3_runtime_split_max_subset_size),
        "phase3_runtime_split_subset_sizes": (
            None
            if getattr(args, "phase3_runtime_split_subset_sizes", None) is None
            else str(args.phase3_runtime_split_subset_sizes)
        ),
        "phase3_runtime_split_child_set_symmetry_policy": str(
            args.phase3_runtime_split_child_set_symmetry_policy
        ),
        "route_a_child_padding_config": route_a_child_padding_config,
        "adapt_child_pool_expansion_mode": str(
            getattr(args, "adapt_child_pool_expansion_mode", "off")
        ),
        "adapt_child_pool_expansion_symmetry_policy": str(
            getattr(args, "adapt_child_pool_expansion_symmetry_policy", "off")
        ),
        "adapt_child_pool_expansion_max_subset_size": int(
            getattr(args, "adapt_child_pool_expansion_max_subset_size", 3)
        ),
        "adapt_child_pool_expansion_subset_sizes": (
            None
            if getattr(args, "adapt_child_pool_expansion_subset_sizes", None) is None
            else str(args.adapt_child_pool_expansion_subset_sizes)
        ),
        "shared_pauli_pool_mode": str(getattr(args, "shared_pauli_pool_mode", "off")),
        "shared_pauli_pool_symmetry_policy": str(
            getattr(args, "shared_pauli_pool_symmetry_policy", "off")
        ),
        "shared_pauli_pool_max_subset_size": int(
            getattr(args, "shared_pauli_pool_max_subset_size", 3)
        ),
        "shared_pauli_pool_subset_sizes": (
            None
            if getattr(args, "shared_pauli_pool_subset_sizes", None) is None
            else str(args.shared_pauli_pool_subset_sizes)
        ),
        "hardware_resolution_mode": str(args.hardware_resolution_mode),
        "gradient_hw_floor": float(args.gradient_hw_floor),
        "gradient_drift_floor": float(args.gradient_drift_floor),
        "hardware_resolution_profile_json": (
            Path(args.hardware_resolution_profile_json)
            if getattr(args, "hardware_resolution_profile_json", None) is not None
            else None
        ),
        "hardware_resolution_profile_name": (
            None
            if getattr(args, "hardware_resolution_profile_name", None) in {None, ""}
            else str(args.hardware_resolution_profile_name)
        ),
        "phase3_selector_policy": str(args.phase3_selector_policy),
        "phase3_selector_geometry_mode": str(args.phase3_selector_geometry_mode),
        "phase3_window_relaxation_mode": str(args.phase3_window_relaxation_mode),
        "phase3_plateau_acquisition_mode": str(args.phase3_plateau_acquisition_mode),
        "phase3_plateau_acquisition_score": str(args.phase3_plateau_acquisition_score),
        "phase3_plateau_unlock_margin": float(args.phase3_plateau_unlock_margin),
        "phase3_plateau_duplicate_policy": str(args.phase3_plateau_duplicate_policy),
        "phase3_plateau_lambda_vol": float(args.phase3_plateau_lambda_vol),
        "phase3_plateau_sigma_min": float(args.phase3_plateau_sigma_min),
        "phase3_plateau_nu_min": float(args.phase3_plateau_nu_min),
        "phase3_plateau_volume_min": float(args.phase3_plateau_volume_min),
        "phase3_plateau_failed_family_patience": int(args.phase3_plateau_failed_family_patience),
        "phase3_plateau_seed_probe_mode": str(args.phase3_plateau_seed_probe_mode),
        "phase3_plateau_seed_probe_count": int(args.phase3_plateau_seed_probe_count),
        "phase3_plateau_seed_probe_radius": float(args.phase3_plateau_seed_probe_radius),
        "phase3_plateau_seed_probe_seed": (
            None
            if args.phase3_plateau_seed_probe_seed is None
            else int(args.phase3_plateau_seed_probe_seed)
        ),
        "phase3_plateau_trial_optimizer": str(args.phase3_plateau_trial_optimizer),
        "phase3_plateau_trial_qngd_maxiter": int(args.phase3_plateau_trial_qngd_maxiter),
        "phase3_shadow_legacy_geometry_mode": str(args.phase3_shadow_legacy_geometry_mode),
        "phase3_shadow_legacy_max_depth": int(args.phase3_shadow_legacy_max_depth),
        "phase3_parent_collapse_debug_max_depth": int(args.phase3_parent_collapse_debug_max_depth),
        "phase3_backend_cost_mode": str(args.phase3_backend_cost_mode),
        "phase3_backend_name": None if args.phase3_backend_name in {None, ""} else str(args.phase3_backend_name),
        "phase3_backend_shortlist": []
                if args.phase3_backend_shortlist in {None, ""}
                else [str(tok).strip() for tok in str(args.phase3_backend_shortlist).split(",") if str(tok).strip() != ""],
        "phase3_backend_transpile_seed": int(args.phase3_backend_transpile_seed),
        "phase3_backend_optimization_level": int(args.phase3_backend_optimization_level),
        "phase3_backend_w_2q": float(args.phase3_backend_w_2q),
        "phase3_backend_w_depth": float(args.phase3_backend_w_depth),
        "phase3_backend_w_size": float(args.phase3_backend_w_size),
        "phase3_selector_debug_topk": int(args.phase3_selector_debug_topk),
        "phase3_selector_debug_max_depth": int(args.phase3_selector_debug_max_depth),
        "phase3_oracle_gradient_config": phase3_oracle_gradient_config,
        "final_noise_audit_config": final_noise_audit_config,
        "phase3_oracle_inner_objective_mode": str(args.phase3_oracle_inner_objective_mode),
    }
