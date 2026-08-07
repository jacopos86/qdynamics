"""Lazy oracle lifecycle helpers for static ADAPT.

This module may construct oracle objects only through caller-supplied binding
dicts. It must not import Qiskit or runtime oracle machinery directly.
"""

from __future__ import annotations

import json
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from pipelines.static_adapt.cli_config import (
    FinalNoiseAuditConfig,
    Phase3OracleGradientConfig,
    SYNTHETIC_COHERENT_1Q_GATES_DEFAULT,
    SYNTHETIC_COHERENT_2Q_GATES_DEFAULT,
    SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT,
    SYNTHETIC_DEPOLARIZING_1Q_GATES_DEFAULT,
    SYNTHETIC_DEPOLARIZING_2Q_GATES_DEFAULT,
    _oracle_mitigation_payload_from_fields,
    _resolve_final_noise_audit_config,
)
from pipelines.static_adapt.noise_routes import (
    _final_noise_audit_config_payload,
    _json_ready,
    _phase3_oracle_mitigation_payload,
    _validate_oracle_execution_request_via_bindings,
)
from src.quantum.ansatz_parameterization import AnsatzParameterLayout, serialize_layout

__all__ = [
    "FinalNoiseAuditSnapshot",
    "Phase3OracleRuntimeContext",
    "_Phase3OracleCleanupGuard",
    "_close_phase3_oracle_resource",
    "_get_phase3_oracle_plan_cached",
    "_normalize_phase3_oracle_backend_info",
    "_run_final_noise_audit",
    "_setup_phase3_oracle_runtime_context",
]


@dataclass(frozen=True)
class FinalNoiseAuditSnapshot:
    h_poly: Any
    parameter_layout: AnsatzParameterLayout
    theta_runtime: tuple[float, ...]
    theta_logical: tuple[float, ...]
    reference_state: np.ndarray
    num_qubits: int
    operator_labels: tuple[str, ...]
    ansatz_depth: int
    runtime_parameter_count: int
    logical_parameter_count: int
    exact_filtered_ground_energy: float
    exact_final_state_energy: float


@dataclass(frozen=True)
class Phase3OracleRuntimeContext:
    bindings: dict[str, Any]
    oracle_obj: Any
    cleanup_guard: "_Phase3OracleCleanupGuard"
    h_qop: Any
    all_z_qop: Any | None
    build_parameterized_ansatz_plan_fn: Callable[..., Any]
    build_runtime_layout_circuit_fn: Callable[..., Any] | None
    backend_info: dict[str, Any] | None
    raw_transport: str | None
    raw_artifact_path: str | None


def _get_phase3_oracle_plan_cached(
    *,
    layout_now: Any,
    build_parameterized_ansatz_plan_fn: Callable[..., Any] | None,
    phase3_oracle_num_qubits: int,
    reference_state: Any,
    plan_cache: dict[str, Any],
) -> Any:
    if build_parameterized_ansatz_plan_fn is None:
        raise RuntimeError("phase3 oracle path is missing its parameterized ansatz plan builder.")
    cache_key = json.dumps(serialize_layout(layout_now), sort_keys=True, separators=(",", ":"))
    cached_plan = plan_cache.get(cache_key)
    if cached_plan is not None:
        return cached_plan
    plan_now = build_parameterized_ansatz_plan_fn(
        layout_now,
        nq=int(phase3_oracle_num_qubits),
        ref_state=np.asarray(reference_state, dtype=complex),
    )
    plan_cache[cache_key] = plan_now
    return plan_now


def _normalize_phase3_oracle_backend_info(
    *,
    config: Phase3OracleGradientConfig | None,
    oracle_obj: Any | None = None,
    raw_bundle: Any | None = None,
) -> dict[str, Any] | None:
    if raw_bundle is not None and config is not None:
        return _json_ready(
            {
                "noise_mode": str(config.noise_mode),
                "estimator_kind": "raw_measurement_oracle",
                "backend_name": raw_bundle.backend_snapshot.get(
                    "backend_name", config.backend_name
                ),
                "using_fake_backend": bool(config.use_fake_backend),
                "details": {
                    "execution_surface": "raw_measurement_v1",
                    "transport": str(raw_bundle.transport),
                    "raw_artifact_path": raw_bundle.raw_artifact_path,
                    "record_count": int(raw_bundle.estimate.record_count),
                    "group_count": int(raw_bundle.estimate.group_count),
                    "term_count": int(raw_bundle.estimate.term_count),
                    "reduction_mode": str(raw_bundle.estimate.reduction_mode),
                    "plan_digest": str(raw_bundle.plan_digest),
                    "structure_digest": str(raw_bundle.structure_digest),
                    "reference_state_digest": raw_bundle.reference_state_digest,
                    "compile_signatures_by_basis": dict(raw_bundle.compile_signatures_by_basis),
                    "backend_snapshot": dict(raw_bundle.backend_snapshot),
                    "transpile_seed": config.seed_transpiler,
                    "seed_transpiler": config.seed_transpiler,
                    "transpile_optimization_level": int(
                        config.transpile_optimization_level
                    ),
                },
            }
        )
    backend_info_raw = getattr(oracle_obj, "backend_info", None) if oracle_obj is not None else None
    if backend_info_raw is not None:
        return _json_ready(getattr(backend_info_raw, "__dict__", backend_info_raw))
    if oracle_obj is not None and config is not None:
        return _json_ready(
            {
                "noise_mode": str(config.noise_mode),
                "estimator_kind": "raw_measurement_oracle",
                "backend_name": config.backend_name,
                "using_fake_backend": bool(config.use_fake_backend),
                "details": {
                    "execution_surface": str(config.execution_surface),
                    "transport": getattr(oracle_obj, "transport", None),
                    "raw_artifact_path": config.raw_artifact_path,
                    "backend_snapshot": dict(getattr(oracle_obj, "backend_snapshot", {}) or {}),
                },
            }
        )
    return None


def _setup_phase3_oracle_runtime_context(
    *,
    config: Phase3OracleGradientConfig,
    h_poly: Any,
    num_qubits: int,
    runtime_bindings_factory: Callable[[], Mapping[str, Any]],
    inner_value_noise_exact_structure_enabled: bool,
    log_fn: Callable[..., None] | None = None,
) -> Phase3OracleRuntimeContext:
    bindings = dict(runtime_bindings_factory())
    build_parameterized_ansatz_plan_fn = bindings["build_parameterized_ansatz_plan"]
    oracle_config = bindings["OracleConfig"](
        noise_mode=str(config.noise_mode),
        shots=int(config.shots),
        seed=int(config.seed),
        seed_transpiler=config.seed_transpiler,
        transpile_optimization_level=int(config.transpile_optimization_level),
        oracle_repeats=int(config.oracle_repeats),
        oracle_aggregate=str(config.oracle_aggregate),
        backend_name=(None if config.backend_name in {None, ""} else str(config.backend_name)),
        use_fake_backend=bool(config.use_fake_backend),
        allow_aer_fallback=True,
        aer_fallback_mode="sampler_shots",
        omp_shm_workaround=True,
        mitigation=dict(_phase3_oracle_mitigation_payload(config)),
        symmetry_mitigation={"mode": "off"},
        execution_surface=str(config.execution_surface),
        raw_transport=str(config.raw_transport),
        raw_store_memory=bool(config.raw_store_memory),
        raw_artifact_path=config.raw_artifact_path,
        value_noise_model=(
            "off"
            if inner_value_noise_exact_structure_enabled
            else str(getattr(config, "value_noise_model", "off"))
        ),
        value_noise_std=(
            0.0
            if inner_value_noise_exact_structure_enabled
            else float(getattr(config, "value_noise_std", 0.0))
        ),
        value_noise_seed=(
            None
            if inner_value_noise_exact_structure_enabled
            else getattr(config, "value_noise_seed", None)
        ),
        value_noise_sigma0_abs=(
            None
            if inner_value_noise_exact_structure_enabled
            else getattr(config, "value_noise_sigma0_abs", None)
        ),
        value_noise_n_eff=(
            None
            if inner_value_noise_exact_structure_enabled
            else getattr(config, "value_noise_n_eff", None)
        ),
        value_noise_semantic=str(
            getattr(
                config,
                "value_noise_semantic",
                "post_expectation_value_noise_not_physical_shots",
            )
        ),
        value_noise_std_source=str(getattr(config, "value_noise_std_source", "explicit_std")),
        synthetic_depolarizing_1q_error=float(
            getattr(config, "synthetic_depolarizing_1q_error", 0.0)
        ),
        synthetic_depolarizing_2q_error=float(
            getattr(config, "synthetic_depolarizing_2q_error", 0.0)
        ),
        synthetic_depolarizing_1q_gates=tuple(
            getattr(
                config,
                "synthetic_depolarizing_1q_gates",
                SYNTHETIC_DEPOLARIZING_1Q_GATES_DEFAULT,
            )
        ),
        synthetic_depolarizing_2q_gates=tuple(
            getattr(
                config,
                "synthetic_depolarizing_2q_gates",
                SYNTHETIC_DEPOLARIZING_2Q_GATES_DEFAULT,
            )
        ),
        synthetic_coherent_1q_angle_std=float(
            getattr(config, "synthetic_coherent_1q_angle_std", 0.0)
        ),
        synthetic_coherent_2q_angle_std=float(
            getattr(config, "synthetic_coherent_2q_angle_std", 0.0)
        ),
        synthetic_coherent_seed=getattr(config, "synthetic_coherent_seed", None),
        synthetic_coherent_generator_mode=str(
            getattr(
                config,
                "synthetic_coherent_generator_mode",
                SYNTHETIC_COHERENT_GENERATOR_MODE_DEFAULT,
            )
        ),
        synthetic_coherent_1q_gates=tuple(
            getattr(config, "synthetic_coherent_1q_gates", SYNTHETIC_COHERENT_1Q_GATES_DEFAULT)
        ),
        synthetic_coherent_2q_gates=tuple(
            getattr(config, "synthetic_coherent_2q_gates", SYNTHETIC_COHERENT_2Q_GATES_DEFAULT)
        ),
    )
    _validate_oracle_execution_request_via_bindings(bindings, oracle_config)
    if (
        str(config.noise_mode).strip().lower() == "backend_scheduled"
        and bool(config.use_fake_backend)
    ):
        try:
            bindings["preflight_backend_scheduled_fake_backend_environment"](oracle_config)
            if log_fn is not None:
                log_fn(
                    "hardcoded_adapt_phase3_oracle_backend_scheduled_preflight_ok",
                    backend_name=oracle_config.backend_name,
                    execution_surface=str(oracle_config.execution_surface),
                )
        except Exception as exc:
            if log_fn is not None:
                log_fn(
                    "hardcoded_adapt_phase3_oracle_backend_scheduled_preflight_failed",
                    backend_name=oracle_config.backend_name,
                    execution_surface=str(oracle_config.execution_surface),
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise

    phase3_oracle_raw_transport = None
    phase3_oracle_raw_artifact_path = None
    phase3_oracle_all_z_qop = None
    build_runtime_layout_circuit_fn = None
    if str(config.execution_surface) == "raw_measurement_v1":
        if str(config.noise_mode).strip().lower() == "runtime":
            oracle_config = bindings["normalize_sampler_raw_runtime_config"](oracle_config)
        oracle_obj = bindings["RawMeasurementOracle"](oracle_config)
        phase3_oracle_all_z_qop = bindings["all_z_full_register_qop"](int(num_qubits))
        phase3_oracle_raw_transport = str(
            getattr(oracle_obj, "transport", config.raw_transport)
        )
        phase3_oracle_raw_artifact_path = config.raw_artifact_path
        backend_info = _normalize_phase3_oracle_backend_info(
            config=config,
            oracle_obj=oracle_obj,
        )
    else:
        oracle_obj = bindings["ExpectationOracle"](oracle_config)
        build_runtime_layout_circuit_fn = bindings["build_runtime_layout_circuit"]
        backend_info = _normalize_phase3_oracle_backend_info(
            config=config,
            oracle_obj=oracle_obj,
        )
    cleanup_guard = _Phase3OracleCleanupGuard(oracle_obj)
    h_qop = bindings["pauli_poly_to_sparse_pauli_op"](h_poly)
    return Phase3OracleRuntimeContext(
        bindings=bindings,
        oracle_obj=oracle_obj,
        cleanup_guard=cleanup_guard,
        h_qop=h_qop,
        all_z_qop=phase3_oracle_all_z_qop,
        build_parameterized_ansatz_plan_fn=build_parameterized_ansatz_plan_fn,
        build_runtime_layout_circuit_fn=build_runtime_layout_circuit_fn,
        backend_info=backend_info,
        raw_transport=phase3_oracle_raw_transport,
        raw_artifact_path=phase3_oracle_raw_artifact_path,
    )


def _close_phase3_oracle_resource(oracle_obj: Any | None) -> None:
    if oracle_obj is None:
        return
    close_oracle = getattr(oracle_obj, "close", None)
    if callable(close_oracle):
        close_oracle()


class _Phase3OracleCleanupGuard:
    def __init__(self, oracle_obj: Any | None) -> None:
        self._oracle_ref = weakref.ref(oracle_obj) if oracle_obj is not None else None
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        oracle_obj = self._oracle_ref() if self._oracle_ref is not None else None
        _close_phase3_oracle_resource(oracle_obj)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _resolve_runtime_bindings(
    *,
    runtime_bindings: Mapping[str, Any] | None,
    runtime_bindings_factory: Callable[[], Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    if runtime_bindings is not None:
        return runtime_bindings
    if runtime_bindings_factory is None:
        raise RuntimeError("final noise audit requires oracle runtime bindings.")
    return runtime_bindings_factory()


def _run_final_noise_audit(
    snapshot: FinalNoiseAuditSnapshot,
    config: FinalNoiseAuditConfig,
    *,
    runtime_bindings: Mapping[str, Any] | None = None,
    runtime_bindings_factory: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    audit_cfg = _resolve_final_noise_audit_config(config)
    bindings = _resolve_runtime_bindings(
        runtime_bindings=runtime_bindings,
        runtime_bindings_factory=runtime_bindings_factory,
    )
    plan = bindings["build_parameterized_ansatz_plan"](
        snapshot.parameter_layout,
        nq=int(snapshot.num_qubits),
        ref_state=np.asarray(snapshot.reference_state, dtype=complex),
    )
    observable = bindings["pauli_poly_to_sparse_pauli_op"](snapshot.h_poly)
    theta_runtime = np.asarray(snapshot.theta_runtime, dtype=float)

    def _evaluate_variant(
        variant_cfg: FinalNoiseAuditConfig,
        *,
        audit_variant: str,
    ) -> dict[str, Any]:
        oracle_config = bindings["OracleConfig"](
            noise_mode=str(variant_cfg.noise_mode),
            shots=int(variant_cfg.shots),
            seed=int(variant_cfg.seed),
            seed_transpiler=variant_cfg.seed_transpiler,
            transpile_optimization_level=int(variant_cfg.transpile_optimization_level),
            oracle_repeats=int(variant_cfg.oracle_repeats),
            oracle_aggregate=str(variant_cfg.oracle_aggregate),
            backend_name=(
                None
                if variant_cfg.backend_name in {None, ""}
                else str(variant_cfg.backend_name)
            ),
            use_fake_backend=bool(variant_cfg.use_fake_backend),
            allow_aer_fallback=True,
            aer_fallback_mode="sampler_shots",
            omp_shm_workaround=True,
            mitigation=dict(
                _oracle_mitigation_payload_from_fields(
                    mitigation_mode=str(variant_cfg.mitigation_mode),
                    local_readout_strategy=variant_cfg.local_readout_strategy,
                    zne_scales=tuple(getattr(variant_cfg, "zne_scales", ()) or ()),
                    dd_sequence=getattr(variant_cfg, "dd_sequence", None),
                    local_gate_twirling=bool(
                        getattr(variant_cfg, "local_gate_twirling", False)
                    ),
                )
            ),
            symmetry_mitigation={"mode": "off"},
            runtime_profile=str(variant_cfg.runtime_profile_name),
            runtime_session=str(variant_cfg.runtime_session_policy),
            execution_surface="expectation_v1",
            value_noise_model=str(getattr(variant_cfg, "value_noise_model", "off")),
            value_noise_std=float(getattr(variant_cfg, "value_noise_std", 0.0)),
            value_noise_seed=getattr(variant_cfg, "value_noise_seed", None),
        )
        validation_report = _validate_oracle_execution_request_via_bindings(bindings, oracle_config)
        normalized_request = (
            None
            if validation_report is None
            else dict(validation_report.get("normalized_request", {}) or {})
        )
        if (
            str(variant_cfg.noise_mode).strip().lower() == "backend_scheduled"
            and bool(variant_cfg.use_fake_backend)
        ):
            bindings["preflight_backend_scheduled_fake_backend_environment"](oracle_config)

        oracle_obj = bindings["ExpectationOracle"](oracle_config)
        close_oracle = getattr(oracle_obj, "close", None)
        try:
            if hasattr(oracle_obj, "evaluate_parameterized"):
                estimate = oracle_obj.evaluate_parameterized(
                    plan=plan,
                    theta_runtime=theta_runtime,
                    observable=observable,
                    runtime_trace_context={
                        "route": "final_noise_audit_v1",
                        "audit_variant": str(audit_variant),
                        "ansatz_depth": int(snapshot.ansatz_depth),
                        "logical_parameter_count": int(snapshot.logical_parameter_count),
                        "runtime_parameter_count": int(snapshot.runtime_parameter_count),
                    },
                )
            else:
                circuit_obj = bindings["build_runtime_layout_circuit"](
                    snapshot.parameter_layout,
                    theta_runtime,
                    int(snapshot.num_qubits),
                    reference_state=np.asarray(snapshot.reference_state, dtype=complex),
                )
                try:
                    setattr(circuit_obj, "_final_noise_audit_route", "final_noise_audit_v1")
                    setattr(circuit_obj, "_final_noise_audit_variant", str(audit_variant))
                except Exception:
                    pass
                estimate = oracle_obj.evaluate(circuit_obj, observable)
            variant_energy = float(getattr(estimate, "mean", 0.0))
            exact_target_delta_e = float(
                variant_energy - float(snapshot.exact_filtered_ground_energy)
            )
            exact_final_state_delta_e = float(
                variant_energy - float(snapshot.exact_final_state_energy)
            )
            return {
                "requested_config": dict(_final_noise_audit_config_payload(variant_cfg) or {}),
                "normalized_request": normalized_request,
                "result": {
                    "requested_estimate_energy": float(variant_energy),
                    "stderr": float(getattr(estimate, "stderr", 0.0) or 0.0),
                    "std": float(getattr(estimate, "std", 0.0) or 0.0),
                    "stdev": float(getattr(estimate, "stdev", 0.0) or 0.0),
                    "n_samples": int(getattr(estimate, "n_samples", 0) or 0),
                    "aggregate": str(getattr(estimate, "aggregate", variant_cfg.oracle_aggregate)),
                    "backend_info": _json_ready(
                        getattr(
                            getattr(oracle_obj, "backend_info", None),
                            "__dict__",
                            getattr(oracle_obj, "backend_info", None),
                        )
                    ),
                },
                "deltas": {
                    "exact_target_delta_e": float(exact_target_delta_e),
                    "exact_target_abs_error": float(abs(exact_target_delta_e)),
                    "exact_final_state_delta_e": float(exact_final_state_delta_e),
                    "exact_final_state_abs_error": float(abs(exact_final_state_delta_e)),
                },
            }
        finally:
            if callable(close_oracle):
                close_oracle()

    requested_eval = _evaluate_variant(audit_cfg, audit_variant="requested")
    output = {
        "status": "completed",
        "strict": bool(audit_cfg.strict),
        "requested_config": dict(requested_eval.get("requested_config", {})),
        "normalized_request": dict(requested_eval.get("normalized_request", {}) or {}),
        "reference": {
            "primary_metric_name": "exact_target_abs_error",
            "exact_filtered_ground_energy": float(snapshot.exact_filtered_ground_energy),
            "exact_final_state_energy": float(snapshot.exact_final_state_energy),
        },
        "snapshot": {
            "ansatz_depth": int(snapshot.ansatz_depth),
            "runtime_parameter_count": int(snapshot.runtime_parameter_count),
            "logical_parameter_count": int(snapshot.logical_parameter_count),
            "operator_labels": [str(x) for x in snapshot.operator_labels],
            "parameterization": serialize_layout(snapshot.parameter_layout),
            "theta_runtime": [float(x) for x in snapshot.theta_runtime],
            "theta_logical": [float(x) for x in snapshot.theta_logical],
        },
        "result": dict(requested_eval.get("result", {})),
        "deltas": dict(requested_eval.get("deltas", {})),
    }

    def _build_unmitigated_baseline_config(
        source_cfg: FinalNoiseAuditConfig,
    ) -> FinalNoiseAuditConfig:
        return _resolve_final_noise_audit_config(
            FinalNoiseAuditConfig(
                noise_mode=str(source_cfg.noise_mode),
                shots=int(source_cfg.shots),
                oracle_repeats=int(source_cfg.oracle_repeats),
                oracle_aggregate=str(source_cfg.oracle_aggregate),
                backend_name=(
                    None
                    if source_cfg.backend_name in {None, ""}
                    else str(source_cfg.backend_name)
                ),
                use_fake_backend=bool(source_cfg.use_fake_backend),
                seed=int(source_cfg.seed),
                mitigation_mode="none",
                local_readout_strategy=None,
                zne_scales=(),
                local_gate_twirling=False,
                dd_sequence=None,
                runtime_profile_name="legacy_runtime_v0",
                runtime_session_policy=str(source_cfg.runtime_session_policy),
                compare_unmitigated_baseline=False,
                seed_transpiler=source_cfg.seed_transpiler,
                transpile_optimization_level=int(source_cfg.transpile_optimization_level),
                strict=bool(source_cfg.strict),
                value_noise_model=str(getattr(source_cfg, "value_noise_model", "off")),
                value_noise_std=float(getattr(source_cfg, "value_noise_std", 0.0)),
                value_noise_seed=getattr(source_cfg, "value_noise_seed", None),
            )
        )

    baseline_requested = bool(audit_cfg.compare_unmitigated_baseline)
    if baseline_requested:
        baseline_cfg = _build_unmitigated_baseline_config(audit_cfg)
        baseline_requested_payload = dict(_final_noise_audit_config_payload(baseline_cfg) or {})
        requested_cfg_payload = dict(output.get("requested_config", {}) or {})
        requested_cfg_payload_cmp = dict(requested_cfg_payload)
        requested_cfg_payload_cmp["compare_unmitigated_baseline"] = False
        if baseline_requested_payload == requested_cfg_payload_cmp:
            output["unmitigated_baseline_comparison"] = {
                "enabled": True,
                "status": "skipped",
                "reason": "requested_matches_unmitigated_baseline",
                "baseline_requested_config": baseline_requested_payload,
            }
        else:
            try:
                baseline_eval = _evaluate_variant(
                    baseline_cfg,
                    audit_variant="unmitigated_baseline",
                )
                requested_energy = float(output.get("result", {}).get("requested_estimate_energy", 0.0))
                baseline_energy = float(
                    baseline_eval.get("result", {}).get("requested_estimate_energy", 0.0)
                )
                requested_exact_target_abs_error = float(
                    output.get("deltas", {}).get("exact_target_abs_error", 0.0)
                )
                baseline_exact_target_abs_error = float(
                    baseline_eval.get("deltas", {}).get("exact_target_abs_error", 0.0)
                )
                requested_exact_final_state_abs_error = float(
                    output.get("deltas", {}).get("exact_final_state_abs_error", 0.0)
                )
                baseline_exact_final_state_abs_error = float(
                    baseline_eval.get("deltas", {}).get("exact_final_state_abs_error", 0.0)
                )
                output["unmitigated_baseline_comparison"] = {
                    "enabled": True,
                    "status": "completed",
                    "baseline_requested_config": dict(
                        baseline_eval.get("requested_config", {})
                    ),
                    "baseline_normalized_request": dict(
                        baseline_eval.get("normalized_request", {}) or {}
                    ),
                    "baseline_result": dict(baseline_eval.get("result", {})),
                    "baseline_deltas": dict(baseline_eval.get("deltas", {})),
                    "comparison_metrics": {
                        "requested_minus_unmitigated_delta_e": float(
                            requested_energy - baseline_energy
                        ),
                        "requested_minus_unmitigated_abs_delta_e": float(
                            abs(requested_energy - baseline_energy)
                        ),
                        "exact_target_abs_error_improvement_vs_unmitigated": float(
                            baseline_exact_target_abs_error
                            - requested_exact_target_abs_error
                        ),
                        "exact_final_state_abs_error_improvement_vs_unmitigated": float(
                            baseline_exact_final_state_abs_error
                            - requested_exact_final_state_abs_error
                        ),
                    },
                }
            except Exception as exc:
                if bool(audit_cfg.strict):
                    raise
                output["unmitigated_baseline_comparison"] = {
                    "enabled": True,
                    "status": "failed",
                    "reason": "evaluation_failed",
                    "baseline_requested_config": baseline_requested_payload,
                    "failure": {
                        "error_type": str(type(exc).__name__),
                        "error_message": str(exc),
                    },
                }
    return output
