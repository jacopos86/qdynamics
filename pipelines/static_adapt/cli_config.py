
import math
from dataclasses import dataclass

from pipelines.exact_bench.noise_oracle_defaults import (
    SYNTHETIC_COHERENT_1Q_GATES_DEFAULT,
    SYNTHETIC_COHERENT_2Q_GATES_DEFAULT,
    SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT,
    SYNTHETIC_DEPOLARIZING_1Q_GATES_DEFAULT,
    SYNTHETIC_DEPOLARIZING_2Q_GATES_DEFAULT,
    normalize_gate_name_tuple,
)
from typing import Any, Callable, Sequence

from pipelines.static_adapt.paper_i_config import PAPER_I_CANONICAL_COST_WEIGHTS
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













