#!/usr/bin/env python3
"""Noise/runtime expectation oracle utilities for HH/Hubbard validation.

This module stays in wrapper/benchmark space. It does not modify core operator
algebra modules and only adapts existing PauliPolynomial + ansatz objects to
Qiskit primitives at the boundary.
"""

from __future__ import annotations

import gzip
import json
import os
import inspect
import subprocess
import sys
import copy
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit import pauli_twirl_2q_gates
from pipelines.qiskit_backend_tools import (
    compile_circuit_for_backend as _compile_circuit_for_backend_shared,
    compiled_gate_stats as _compiled_gate_stats_shared,
    list_local_fake_backend_names as _list_local_fake_backend_names_shared,
    load_local_fake_backend as _load_local_fake_backend_shared,
    safe_circuit_depth as _safe_circuit_depth_shared,
    snapshot_backend_target as _snapshot_backend_target_shared,
)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from src.quantum.ansatz_parameterization import AnsatzParameterLayout


def _oracle_debug_log(event: str, /, **payload: Any) -> None:
    record = {"event": str(event), **payload, "ts_utc": datetime.now(timezone.utc).isoformat()}
    print(f"AI_LOG {json.dumps(record, sort_keys=True, default=str)}", flush=True)


def _oracle_compile_heartbeat(
    event: str,
    /,
    *,
    noise_mode: str,
    backend_name: str | None,
    transpile_seed: int,
    optimization_level: int,
    cache_hit: bool,
    circuit: QuantumCircuit,
    elapsed_s: float | None = None,
    compiled_payload: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "noise_mode": str(noise_mode),
        "backend_name": (None if backend_name in {None, ""} else str(backend_name)),
        "transpile_seed": int(transpile_seed),
        "optimization_level": int(optimization_level),
        "cache_hit": bool(cache_hit),
        "circuit_name": str(getattr(circuit, "name", "") or ""),
        "circuit_num_qubits": int(getattr(circuit, "num_qubits", 0)),
        "circuit_num_clbits": int(getattr(circuit, "num_clbits", 0)),
        "circuit_num_parameters": int(len(tuple(getattr(circuit, "parameters", ())))),
    }
    if elapsed_s is not None:
        payload["elapsed_s"] = float(elapsed_s)
    if compiled_payload is not None:
        logical_to_physical = compiled_payload.get("logical_to_physical", ())
        payload["layout_physical_qubits"] = [int(x) for x in logical_to_physical]
        payload["compiled_num_qubits"] = int(
            compiled_payload.get(
                "compiled_num_qubits",
                getattr(compiled_payload.get("compiled", None), "num_qubits", getattr(circuit, "num_qubits", 0)),
            )
        )
        compiled = compiled_payload.get("compiled", None)
        if compiled is not None:
            payload.update(_compiled_metrics_payload(compiled))
    _oracle_debug_log(event, **payload)


@dataclass(frozen=True)
class OracleConfig:
    noise_mode: str = "ideal"  # ideal | shots | aer_noise | runtime | backend_scheduled
    shots: int = 2048
    seed: int = 7
    seed_transpiler: int | None = None
    transpile_optimization_level: int = 1
    oracle_repeats: int = 1
    oracle_aggregate: str = "mean"  # mean | median
    backend_name: str | None = None
    use_fake_backend: bool = False
    approximation: bool = False
    abelian_grouping: bool = True
    allow_aer_fallback: bool = True
    aer_fallback_mode: str = "sampler_shots"
    omp_shm_workaround: bool = True
    mitigation: dict[str, Any] | str = "none"
    symmetry_mitigation: dict[str, Any] | str = "off"
    runtime_profile: dict[str, Any] | str = "legacy_runtime_v0"
    runtime_session: dict[str, Any] | str = "prefer_session"
    execution_surface: str = "expectation_v1"
    raw_transport: str = "auto"
    raw_store_memory: bool = False
    raw_grouping_mode: str = "qwc_basis_cover_reuse"
    raw_artifact_path: str | None = None


@dataclass(frozen=True)
class MitigationConfig:
    mode: str = "none"  # none | readout | zne | dd
    zne_scales: tuple[float, ...] = ()
    dd_sequence: str | None = None
    local_readout_strategy: str | None = None
    local_gate_twirling: bool | None = None


@dataclass(frozen=True)
class RuntimeEstimatorProfileConfig:
    name: str = "legacy_runtime_v0"
    resilience_level: int | None = None
    default_shots: int | None = None
    default_precision: float | None = None
    max_execution_time: int | None = None
    init_qubits: bool | None = True
    measure_mitigation: bool | None = None
    measure_twirling: bool | None = None
    gate_twirling: bool | None = None
    twirling_strategy: str | None = None
    zne_mitigation: bool | None = None
    zne_noise_factors: tuple[float, ...] = ()
    zne_extrapolator: tuple[str, ...] = ()
    pec_mitigation: bool | None = None
    dd_enable: bool | None = None
    dd_sequence: str | None = None


@dataclass(frozen=True)
class RuntimeSessionPolicyConfig:
    mode: str = "prefer_session"  # prefer_session | require_session | backend_only


@dataclass(frozen=True)
class OracleEstimate:
    mean: float
    std: float
    stdev: float
    stderr: float
    n_samples: int
    raw_values: list[float]
    aggregate: str


@dataclass(frozen=True)
class RawObservableEstimate:
    mean: float
    std: float
    stdev: float
    stderr: float
    n_samples: int
    raw_values: tuple[float, ...]
    aggregate: str
    total_shots: int
    group_count: int
    term_count: int
    record_count: int
    reduction_mode: str


@dataclass(frozen=True)
class RawMeasurementRecord:
    schema_version: str
    record_id: str
    evaluation_id: str
    execution_surface: str
    observable_family: str
    semantic_tags: dict[str, Any]
    group_index: int
    basis_label: str
    group_terms: tuple[dict[str, Any], ...]
    plan_digest: str
    structure_digest: str
    reference_state_digest: str | None
    theta_runtime: tuple[float, ...]
    theta_digest: str
    logical_parameter_count: int
    runtime_parameter_count: int
    num_qubits: int
    measured_logical_qubits: tuple[int, ...]
    measured_physical_qubits: tuple[int, ...] | None
    logical_to_physical: tuple[int, ...] | None
    physical_to_logical: dict[str, int] | None
    compile_signature: dict[str, Any]
    backend_snapshot: dict[str, Any]
    transport: str
    call_path: str
    job_records: tuple[dict[str, Any], ...]
    repeat_index: int
    shots_requested: int
    shots_completed: int
    counts: dict[str, int]
    requested_mitigation: dict[str, Any]
    requested_symmetry_mitigation: dict[str, Any]
    requested_runtime_profile: dict[str, Any]
    requested_runtime_session: dict[str, Any]
    parent_record_ids: tuple[str, ...]
    emitted_utc: str


@dataclass(frozen=True)
class RawExecutionBundle:
    schema_version: str
    evaluation_id: str
    execution_surface: str
    observable_family: str
    transport: str
    estimate: RawObservableEstimate
    records: tuple[RawMeasurementRecord, ...]
    raw_artifact_path: str | None
    plan_digest: str
    structure_digest: str
    reference_state_digest: str | None
    compile_signatures_by_basis: dict[str, dict[str, Any]]
    backend_snapshot: dict[str, Any]


@dataclass(frozen=True)
class SymmetryMitigationConfig:
    mode: str = "off"  # off | verify_only | postselect_diag_v1 | projector_renorm_v1
    num_sites: int | None = None
    ordering: str = "blocked"
    sector_n_up: int | None = None
    sector_n_dn: int | None = None


_MITIGATION_MODES = {"none", "readout", "zne", "dd"}
_SYMMETRY_MITIGATION_MODES = {"off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"}
_LOCAL_READOUT_STRATEGIES = {"mthree"}
_RUNTIME_PROFILE_NAMES = {
    "legacy_runtime_v0",
    "main_twirled_readout_v1",
    "dd_probe_twirled_readout_v1",
    "final_audit_zne_twirled_readout_v1",
}
_RUNTIME_SESSION_POLICY_MODES = {"prefer_session", "require_session", "backend_only"}


def _parse_zne_scales(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, str):
        tokens = [tok.strip() for tok in str(raw).split(",")]
        vals = [tok for tok in tokens if tok]
    elif isinstance(raw, Sequence):
        vals = [str(v).strip() for v in list(raw)]
        vals = [tok for tok in vals if tok]
    else:
        vals = [str(raw).strip()]
    out: list[float] = []
    for tok in vals:
        value = float(tok)
        if (not np.isfinite(value)) or (value <= 0.0):
            raise ValueError(f"Invalid mitigation zne scale {tok!r}; expected finite > 0.")
        out.append(float(value))
    return out


def _coerce_optional_bool(raw: Any, *, field_name: str) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, (bool, np.bool_)):
        return bool(raw)
    if isinstance(raw, (int, np.integer)):
        return bool(raw)
    token = str(raw).strip().lower()
    if token in {"", "none", "null"}:
        return None
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Unsupported boolean value for {field_name}: {raw!r}; expected true/false."
    )


def normalize_mitigation_config(mitigation: Any) -> dict[str, Any]:
    mode = "none"
    zne_scales: list[float] = []
    dd_sequence: str | None = None
    local_readout_strategy: str | None = None
    local_gate_twirling: bool | None = None

    if mitigation is None:
        pass
    elif isinstance(mitigation, MitigationConfig):
        mode = str(mitigation.mode).strip().lower() or "none"
        zne_scales = _parse_zne_scales(list(mitigation.zne_scales))
        dd_sequence = None if mitigation.dd_sequence is None else str(mitigation.dd_sequence)
        local_readout_strategy = (
            None
            if mitigation.local_readout_strategy is None
            else str(mitigation.local_readout_strategy).strip().lower() or None
        )
        local_gate_twirling = _coerce_optional_bool(
            mitigation.local_gate_twirling,
            field_name="local_gate_twirling",
        )
    elif isinstance(mitigation, str):
        mode = str(mitigation).strip().lower() or "none"
    elif isinstance(mitigation, Mapping):
        mode = str(mitigation.get("mode", mitigation.get("mitigation", "none"))).strip().lower() or "none"
        zne_raw = mitigation.get("zne_scales", mitigation.get("zneScales", []))
        zne_scales = _parse_zne_scales(zne_raw)
        dd_raw = mitigation.get("dd_sequence", mitigation.get("ddSequence", None))
        dd_sequence = None if dd_raw is None else str(dd_raw)
        local_raw = mitigation.get(
            "local_readout_strategy",
            mitigation.get("localReadoutStrategy", mitigation.get("strategy", None)),
        )
        local_readout_strategy = (
            None if local_raw is None else str(local_raw).strip().lower() or None
        )
        local_gate_twirling = _coerce_optional_bool(
            mitigation.get(
                "local_gate_twirling",
                mitigation.get("localGateTwirling", mitigation.get("gate_twirling", mitigation.get("gateTwirling", None))),
            ),
            field_name="local_gate_twirling",
        )
    else:
        raise ValueError(
            "Unsupported mitigation config type; expected str, dict, MitigationConfig, or None."
        )

    if mode not in _MITIGATION_MODES:
        raise ValueError(
            f"Unsupported mitigation mode {mode!r}; expected one of {sorted(_MITIGATION_MODES)}."
        )
    if mode != "readout":
        local_readout_strategy = None
    if local_readout_strategy is not None and local_readout_strategy not in _LOCAL_READOUT_STRATEGIES:
        raise ValueError(
            "Unsupported local readout strategy "
            f"{local_readout_strategy!r}; expected one of {sorted(_LOCAL_READOUT_STRATEGIES)}."
        )

    out = {
        "mode": str(mode),
        "zne_scales": [float(x) for x in zne_scales],
        "dd_sequence": dd_sequence,
        "local_readout_strategy": local_readout_strategy,
    }
    if bool(local_gate_twirling):
        out["local_gate_twirling"] = True
    return out


def _parse_string_tuple(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        vals = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
    elif isinstance(raw, Sequence):
        vals = [str(v).strip() for v in raw if str(v).strip()]
    else:
        vals = [str(raw).strip()]
    return [str(v) for v in vals if str(v)]


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def normalize_runtime_estimator_profile_config(profile: Any) -> dict[str, Any]:
    raw_name = "legacy_runtime_v0"
    overrides: dict[str, Any] = {}
    if profile is None:
        pass
    elif isinstance(profile, RuntimeEstimatorProfileConfig):
        raw_name = str(profile.name).strip().lower() or "legacy_runtime_v0"
        overrides = {
            "resilience_level": profile.resilience_level,
            "default_shots": profile.default_shots,
            "default_precision": profile.default_precision,
            "max_execution_time": profile.max_execution_time,
            "init_qubits": profile.init_qubits,
            "measure_mitigation": profile.measure_mitigation,
            "measure_twirling": profile.measure_twirling,
            "gate_twirling": profile.gate_twirling,
            "twirling_strategy": profile.twirling_strategy,
            "zne_mitigation": profile.zne_mitigation,
            "zne_noise_factors": list(profile.zne_noise_factors),
            "zne_extrapolator": list(profile.zne_extrapolator),
            "pec_mitigation": profile.pec_mitigation,
            "dd_enable": profile.dd_enable,
            "dd_sequence": profile.dd_sequence,
        }
    elif isinstance(profile, str):
        raw_name = str(profile).strip().lower() or "legacy_runtime_v0"
    elif isinstance(profile, Mapping):
        raw_name = str(profile.get("name", profile.get("profile", "legacy_runtime_v0"))).strip().lower() or "legacy_runtime_v0"
        overrides = dict(profile)
    else:
        raise ValueError(
            "Unsupported runtime profile config type; expected str, dict, RuntimeEstimatorProfileConfig, or None."
        )

    if raw_name not in _RUNTIME_PROFILE_NAMES:
        raise ValueError(
            f"Unsupported runtime profile {raw_name!r}; expected one of {sorted(_RUNTIME_PROFILE_NAMES)}."
        )

    if raw_name == "legacy_runtime_v0":
        cfg: dict[str, Any] = {
            "name": str(raw_name),
            "resilience_level": None,
            "default_shots": None,
            "default_precision": None,
            "max_execution_time": None,
            "init_qubits": None,
            "measure_mitigation": None,
            "measure_twirling": None,
            "gate_twirling": None,
            "twirling_strategy": None,
            "zne_mitigation": None,
            "zne_noise_factors": [],
            "zne_extrapolator": [],
            "pec_mitigation": None,
            "dd_enable": None,
            "dd_sequence": None,
        }
    elif raw_name == "main_twirled_readout_v1":
        cfg = {
            "name": str(raw_name),
            "resilience_level": 1,
            "default_shots": None,
            "default_precision": None,
            "max_execution_time": None,
            "init_qubits": True,
            "measure_mitigation": True,
            "measure_twirling": True,
            "gate_twirling": True,
            "twirling_strategy": "active-accum",
            "zne_mitigation": False,
            "zne_noise_factors": [],
            "zne_extrapolator": [],
            "pec_mitigation": False,
            "dd_enable": False,
            "dd_sequence": None,
        }
    elif raw_name == "dd_probe_twirled_readout_v1":
        cfg = {
            "name": str(raw_name),
            "resilience_level": 1,
            "default_shots": None,
            "default_precision": None,
            "max_execution_time": None,
            "init_qubits": True,
            "measure_mitigation": True,
            "measure_twirling": True,
            "gate_twirling": False,
            "twirling_strategy": "active-accum",
            "zne_mitigation": False,
            "zne_noise_factors": [],
            "zne_extrapolator": [],
            "pec_mitigation": False,
            "dd_enable": True,
            "dd_sequence": "XpXm",
        }
    else:
        cfg = {
            "name": str(raw_name),
            "resilience_level": 1,
            "default_shots": None,
            "default_precision": None,
            "max_execution_time": None,
            "init_qubits": True,
            "measure_mitigation": True,
            "measure_twirling": True,
            "gate_twirling": True,
            "twirling_strategy": "active-accum",
            "zne_mitigation": True,
            "zne_noise_factors": [1.0, 3.0, 5.0],
            "zne_extrapolator": ["exponential", "linear"],
            "pec_mitigation": False,
            "dd_enable": False,
            "dd_sequence": None,
        }

    if overrides:
        if "resilience_level" in overrides and overrides.get("resilience_level", None) is not None:
            cfg["resilience_level"] = int(overrides["resilience_level"])
        if "default_shots" in overrides and overrides.get("default_shots", None) is not None:
            cfg["default_shots"] = int(overrides["default_shots"])
        if "default_precision" in overrides and overrides.get("default_precision", None) is not None:
            cfg["default_precision"] = float(overrides["default_precision"])
        if "max_execution_time" in overrides and overrides.get("max_execution_time", None) is not None:
            cfg["max_execution_time"] = int(overrides["max_execution_time"])
        for key in (
            "init_qubits",
            "measure_mitigation",
            "measure_twirling",
            "gate_twirling",
            "zne_mitigation",
            "pec_mitigation",
            "dd_enable",
        ):
            if key in overrides and overrides.get(key, None) is not None:
                cfg[key] = bool(overrides[key])
        if "twirling_strategy" in overrides and overrides.get("twirling_strategy", None) not in {None, ""}:
            cfg["twirling_strategy"] = str(overrides["twirling_strategy"])
        if "dd_sequence" in overrides and overrides.get("dd_sequence", None) not in {None, ""}:
            cfg["dd_sequence"] = str(overrides["dd_sequence"])
        if "zne_noise_factors" in overrides:
            cfg["zne_noise_factors"] = _parse_zne_scales(overrides.get("zne_noise_factors", None))
        if "zne_extrapolator" in overrides:
            cfg["zne_extrapolator"] = _parse_string_tuple(overrides.get("zne_extrapolator", None))

    if not bool(cfg.get("zne_mitigation", False)):
        cfg["zne_noise_factors"] = []
        cfg["zne_extrapolator"] = []
    if not bool(cfg.get("dd_enable", False)):
        cfg["dd_sequence"] = None
    return cfg


def normalize_runtime_session_policy_config(session_policy: Any) -> dict[str, Any]:
    mode = "prefer_session"
    if session_policy is None:
        pass
    elif isinstance(session_policy, RuntimeSessionPolicyConfig):
        mode = str(session_policy.mode).strip().lower() or "prefer_session"
    elif isinstance(session_policy, str):
        mode = str(session_policy).strip().lower() or "prefer_session"
    elif isinstance(session_policy, Mapping):
        mode = str(session_policy.get("mode", session_policy.get("session_policy", "prefer_session"))).strip().lower() or "prefer_session"
    else:
        raise ValueError(
            "Unsupported runtime session policy config type; expected str, dict, RuntimeSessionPolicyConfig, or None."
        )
    if mode not in _RUNTIME_SESSION_POLICY_MODES:
        raise ValueError(
            f"Unsupported runtime session policy {mode!r}; expected one of {sorted(_RUNTIME_SESSION_POLICY_MODES)}."
        )
    return {"mode": str(mode)}


def normalize_sampler_raw_runtime_config(config: OracleConfig) -> OracleConfig:
    normalized = OracleConfig(
        noise_mode=str(getattr(config, "noise_mode", "runtime")).strip().lower() or "runtime",
        shots=int(getattr(config, "shots", 2048)),
        seed=int(getattr(config, "seed", 7)),
        seed_transpiler=(
            None
            if getattr(config, "seed_transpiler", None) is None
            else int(getattr(config, "seed_transpiler"))
        ),
        transpile_optimization_level=int(getattr(config, "transpile_optimization_level", 1)),
        oracle_repeats=max(1, int(getattr(config, "oracle_repeats", 1))),
        oracle_aggregate=str(getattr(config, "oracle_aggregate", "mean")).strip().lower() or "mean",
        backend_name=(
            None
            if getattr(config, "backend_name", None) in {None, ""}
            else str(getattr(config, "backend_name"))
        ),
        use_fake_backend=bool(getattr(config, "use_fake_backend", False)),
        approximation=bool(getattr(config, "approximation", False)),
        abelian_grouping=bool(getattr(config, "abelian_grouping", True)),
        allow_aer_fallback=bool(getattr(config, "allow_aer_fallback", True)),
        aer_fallback_mode=str(getattr(config, "aer_fallback_mode", "sampler_shots")).strip().lower()
        or "sampler_shots",
        omp_shm_workaround=bool(getattr(config, "omp_shm_workaround", True)),
        mitigation=dict(normalize_mitigation_config(getattr(config, "mitigation", "none"))),
        symmetry_mitigation=dict(
            normalize_symmetry_mitigation_config(getattr(config, "symmetry_mitigation", "off"))
        ),
        runtime_profile=dict(
            normalize_runtime_estimator_profile_config(
                getattr(config, "runtime_profile", "legacy_runtime_v0")
            )
        ),
        runtime_session=dict(
            normalize_runtime_session_policy_config(
                getattr(config, "runtime_session", "prefer_session")
            )
        ),
        execution_surface=str(
            getattr(config, "execution_surface", "raw_measurement_v1")
        ).strip().lower()
        or "raw_measurement_v1",
        raw_transport=str(getattr(config, "raw_transport", "auto")).strip().lower() or "auto",
        raw_store_memory=bool(getattr(config, "raw_store_memory", False)),
        raw_grouping_mode=str(
            getattr(config, "raw_grouping_mode", "qwc_basis_cover_reuse")
        ).strip().lower()
        or "qwc_basis_cover_reuse",
        raw_artifact_path=(
            None
            if getattr(config, "raw_artifact_path", None) in {None, ""}
            else str(getattr(config, "raw_artifact_path"))
        ),
    )
    mitigation_cfg = dict(normalize_mitigation_config(normalized.mitigation))
    symmetry_cfg = dict(normalize_symmetry_mitigation_config(normalized.symmetry_mitigation))
    runtime_profile_cfg = dict(
        normalize_runtime_estimator_profile_config(normalized.runtime_profile)
    )
    noise_mode = str(normalized.noise_mode).strip().lower()
    execution_surface = str(normalized.execution_surface).strip().lower()
    raw_transport = str(normalized.raw_transport).strip().lower()

    if noise_mode != "runtime":
        raise ValueError("sampler raw runtime config requires noise_mode='runtime'.")
    if bool(normalized.use_fake_backend):
        raise ValueError("sampler raw runtime config requires use_fake_backend=False.")
    if normalized.backend_name in {None, ""}:
        raise ValueError("sampler raw runtime config requires backend_name.")
    if execution_surface != "raw_measurement_v1":
        raise ValueError(
            "sampler raw runtime config requires execution_surface='raw_measurement_v1'."
        )
    if str(mitigation_cfg.get("mode", "none")) != "none":
        raise ValueError("sampler raw runtime config requires mitigation_mode='none'.")
    if str(symmetry_cfg.get("mode", "off")) != "off":
        raise ValueError("sampler raw runtime config requires symmetry_mitigation='off'.")
    if str(runtime_profile_cfg.get("name", "legacy_runtime_v0")) != "legacy_runtime_v0":
        raise ValueError(
            "sampler raw runtime config requires runtime_profile='legacy_runtime_v0'."
        )
    if raw_transport not in {"auto", "sampler_v2"}:
        raise ValueError(
            "sampler raw runtime config requires raw_transport in {'auto','sampler_v2'}."
        )
    return normalized


def normalize_symmetry_mitigation_config(symmetry_mitigation: Any) -> dict[str, Any]:
    mode = "off"
    num_sites: int | None = None
    ordering = "blocked"
    sector_n_up: int | None = None
    sector_n_dn: int | None = None

    if symmetry_mitigation is None:
        pass
    elif isinstance(symmetry_mitigation, SymmetryMitigationConfig):
        mode = str(symmetry_mitigation.mode).strip().lower() or "off"
        num_sites = (
            None if symmetry_mitigation.num_sites is None else int(symmetry_mitigation.num_sites)
        )
        ordering = str(symmetry_mitigation.ordering).strip().lower() or "blocked"
        sector_n_up = (
            None if symmetry_mitigation.sector_n_up is None else int(symmetry_mitigation.sector_n_up)
        )
        sector_n_dn = (
            None if symmetry_mitigation.sector_n_dn is None else int(symmetry_mitigation.sector_n_dn)
        )
    elif isinstance(symmetry_mitigation, str):
        mode = str(symmetry_mitigation).strip().lower() or "off"
    elif isinstance(symmetry_mitigation, Mapping):
        mode = str(
            symmetry_mitigation.get("mode", symmetry_mitigation.get("symmetry_mitigation", "off"))
        ).strip().lower() or "off"
        num_sites_raw = symmetry_mitigation.get("num_sites", symmetry_mitigation.get("L", None))
        ordering_raw = symmetry_mitigation.get("ordering", "blocked")
        n_up_raw = symmetry_mitigation.get("sector_n_up", symmetry_mitigation.get("n_up", None))
        n_dn_raw = symmetry_mitigation.get("sector_n_dn", symmetry_mitigation.get("n_dn", None))
        num_sites = None if num_sites_raw is None else int(num_sites_raw)
        ordering = str(ordering_raw).strip().lower() or "blocked"
        sector_n_up = None if n_up_raw is None else int(n_up_raw)
        sector_n_dn = None if n_dn_raw is None else int(n_dn_raw)
    else:
        raise ValueError(
            "Unsupported symmetry mitigation config type; expected str, dict, SymmetryMitigationConfig, or None."
        )

    if mode not in _SYMMETRY_MITIGATION_MODES:
        raise ValueError(
            f"Unsupported symmetry mitigation mode {mode!r}; expected one of {sorted(_SYMMETRY_MITIGATION_MODES)}."
        )

    return {
        "mode": str(mode),
        "num_sites": (None if num_sites is None else int(num_sites)),
        "ordering": str(ordering),
        "sector_n_up": (None if sector_n_up is None else int(sector_n_up)),
        "sector_n_dn": (None if sector_n_dn is None else int(sector_n_dn)),
    }


def normalize_ideal_reference_symmetry_mitigation(
    symmetry_mitigation: Any,
    *,
    noise_mode: str,
) -> dict[str, Any]:
    cfg = normalize_symmetry_mitigation_config(symmetry_mitigation)
    if str(noise_mode).strip().lower() == "runtime" and str(cfg.get("mode", "off")) not in {"off", "verify_only"}:
        downgraded = dict(cfg)
        downgraded["mode"] = "verify_only"
        return downgraded
    return cfg


@dataclass(frozen=True)
class NoiseBackendInfo:
    noise_mode: str
    estimator_kind: str
    backend_name: str | None = None
    using_fake_backend: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeJobRecord:
    job_id: str | None
    repeat_index: int
    call_path: str
    status: str
    created_utc: str | None = None
    running_utc: str | None = None
    completed_utc: str | None = None
    expectation_value: float | None = None
    error: str | None = None
    backend_name: str | None = None
    session_id: str | None = None
    usage_quantum_seconds: float | None = None


@dataclass(frozen=True)
class EstimatorExecutionResult:
    expectation_value: float
    job_records: tuple[RuntimeJobRecord, ...]
    used_call_path: str


@dataclass(frozen=True)
class SamplerExecutionResult:
    counts: dict[str, int]
    job_records: tuple[RuntimeJobRecord, ...]
    used_call_path: str


class SubmittedRuntimeJobError(RuntimeError):
    """A Runtime job was submitted, then failed before returning an estimate."""

    def __init__(
        self,
        message: str,
        *,
        record: RuntimeJobRecord,
        original: Exception,
    ) -> None:
        super().__init__(message)
        self.record = record
        self.original = original


class LocalDDSupportError(RuntimeError):
    """Local backend_scheduled DD could not be applied honestly."""


def _iter_exception_chain(exc: BaseException | None) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current = exc
    while isinstance(current, BaseException) and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        next_exc = current.__cause__ if current.__cause__ is not None else current.__context__
        current = next_exc if isinstance(next_exc, BaseException) else None
    return chain


def _is_invalid_t2_relaxation_error(exc: BaseException) -> bool:
    required = (
        "Invalid T_2 relaxation time parameter",
        "T_2 greater than 2 * T_1",
    )
    for item in _iter_exception_chain(exc):
        text = str(item)
        if all(token in text for token in required):
            return True
    return False


class _BackendPropertiesOverrideWrapper:
    """BackendV1-style wrapper exposing overridden BackendProperties only."""

    def __init__(self, backend: Any, properties: Any) -> None:
        self._backend = backend
        self._properties = properties
        self.version = 1

    def properties(self) -> Any:
        return self._properties

    def configuration(self) -> Any:
        return self._backend.configuration()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


class _RelaxationPropertiesProxy:
    """Minimal BackendProperties-like proxy for Aer V1 noise construction."""

    _TIME_UNIT_SCALE = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
        "ps": 1e-12,
    }
    _FREQ_UNIT_SCALE = {
        "hz": 1.0,
        "khz": 1e3,
        "mhz": 1e6,
        "ghz": 1e9,
        "thz": 1e12,
    }

    def __init__(self, properties: Any) -> None:
        self._properties = properties
        self.qubits = copy.deepcopy(getattr(properties, "qubits", []))
        self.gates = copy.deepcopy(getattr(properties, "gates", []))

    def _find_qubit_item(self, qubit: int, name: str) -> Any | None:
        if qubit < 0 or qubit >= len(self.qubits):
            return None
        for item in list(self.qubits[int(qubit)]):
            if getattr(item, "name", None) == str(name):
                return item
        return None

    @classmethod
    def _scaled_value(cls, item: Any | None, unit_scale: Mapping[str, float]) -> float | None:
        if item is None or not hasattr(item, "value"):
            return None
        try:
            value = float(getattr(item, "value"))
        except Exception:
            return None
        unit = str(getattr(item, "unit", "") or "").strip().lower()
        if unit:
            value *= float(unit_scale.get(unit, 1.0))
        return float(value)

    def t1(self, qubit: int) -> float | None:
        return self._scaled_value(self._find_qubit_item(int(qubit), "T1"), self._TIME_UNIT_SCALE)

    def t2(self, qubit: int) -> float | None:
        return self._scaled_value(self._find_qubit_item(int(qubit), "T2"), self._TIME_UNIT_SCALE)

    def frequency(self, qubit: int) -> float | None:
        return self._scaled_value(
            self._find_qubit_item(int(qubit), "frequency"),
            self._FREQ_UNIT_SCALE,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._properties, name)


def _sanitize_backend_relaxation_properties_for_local_dd(
    backend: Any,
) -> tuple[Any, dict[str, Any]]:
    if backend is None or not hasattr(backend, "properties"):
        raise LocalDDSupportError("backend_scheduled backend properties are unavailable for local DD retry.")
    try:
        properties = _RelaxationPropertiesProxy(backend.properties())
    except Exception as exc:
        raise LocalDDSupportError(
            f"Failed to copy backend properties for local DD retry: {type(exc).__name__}: {exc}"
        ) from exc
    qubits = getattr(properties, "qubits", None)
    if qubits is None:
        raise LocalDDSupportError("backend properties do not expose qubit relaxation metadata.")

    affected_qubits: list[int] = []
    for qubit_index, qubit_props in enumerate(list(qubits)):
        t1_item = None
        t2_item = None
        for item in list(qubit_props):
            name = getattr(item, "name", None)
            if name == "T1":
                t1_item = item
            elif name == "T2":
                t2_item = item
        if t1_item is None or t2_item is None:
            continue
        try:
            t1_value = float(getattr(t1_item, "value"))
            t2_value = float(getattr(t2_item, "value"))
        except Exception:
            continue
        if (not np.isfinite(t1_value)) or (not np.isfinite(t2_value)) or t1_value <= 0.0:
            continue
        upper = 2.0 * float(t1_value)
        if float(t2_value) > float(upper):
            t2_item.value = float(np.nextafter(float(upper), 0.0))
            affected_qubits.append(int(qubit_index))

    return properties, {
        "applied": bool(affected_qubits),
        "strategy": "clamp_t2_to_2t1",
        "affected_qubits": [int(q) for q in affected_qubits],
        "retry_trigger": "invalid_t2_gt_2t1",
    }


def _rewrite_local_dd_measurement_basis_ops(circuit: QuantumCircuit) -> QuantumCircuit:
    """Rewrite DD-path measurement basis changes into timed target-friendly gates."""
    rewritten = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
    rewritten.global_phase = circuit.global_phase
    for inst in circuit.data:
        op = inst.operation
        qargs = list(inst.qubits)
        cargs = list(inst.clbits)
        if len(qargs) == 1 and not cargs:
            if op.name == "h":
                rewritten.rz(np.pi / 2.0, qargs[0])
                rewritten.sx(qargs[0])
                rewritten.rz(np.pi / 2.0, qargs[0])
                continue
            if op.name == "sdg":
                rewritten.rz(-np.pi / 2.0, qargs[0])
                continue
        rewritten.append(op, qargs, cargs)
    return rewritten


def _to_ixyz(label_exyz: str) -> str:
    return (
        str(label_exyz)
        .replace("e", "I")
        .replace("x", "X")
        .replace("y", "Y")
        .replace("z", "Z")
    )


_PAULI_POLY_TO_QOP_MATH = "H = sum_j c_j P_j  ->  SparsePauliOp([(P_j, c_j)])"


def _pauli_poly_to_sparse_pauli_op(poly: Any, tol: float = 1e-12) -> SparsePauliOp:
    """Convert repo PauliPolynomial (exyz labels) into SparsePauliOp."""
    terms = list(poly.return_polynomial())
    if not terms:
        return SparsePauliOp.from_list([("I", 0.0)])

    nq = int(terms[0].nqubit())
    coeff_map: dict[str, complex] = {}
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        lbl = _to_ixyz(str(term.pw2strng()))
        coeff_map[lbl] = coeff_map.get(lbl, 0.0 + 0.0j) + coeff

    cleaned = [(lbl, c) for lbl, c in coeff_map.items() if abs(c) > float(tol)]
    if not cleaned:
        cleaned = [("I" * nq, 0.0 + 0.0j)]
    return SparsePauliOp.from_list(cleaned).simplify(atol=float(tol))


def _ansatz_terms_with_parameters(ansatz: Any, theta: np.ndarray) -> list[tuple[Any, float]]:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    num_parameters = int(getattr(ansatz, "num_parameters", -1))
    if num_parameters < 0:
        raise ValueError("ansatz is missing num_parameters")
    if int(theta.size) != num_parameters:
        raise ValueError(f"theta length {int(theta.size)} does not match ansatz.num_parameters={num_parameters}")

    reps = int(getattr(ansatz, "reps", 1))
    out: list[tuple[Any, float]] = []
    k = 0

    layer_term_groups = getattr(ansatz, "layer_term_groups", None)
    if isinstance(layer_term_groups, list) and layer_term_groups:
        for _ in range(reps):
            for _name, terms in layer_term_groups:
                val = float(theta[k])
                for term in terms:
                    out.append((term.polynomial, val))
                k += 1
    else:
        base_terms = list(getattr(ansatz, "base_terms", []))
        if not base_terms:
            raise ValueError("ansatz has no base_terms/layer_term_groups")
        for _ in range(reps):
            for term in base_terms:
                out.append((term.polynomial, float(theta[k])))
                k += 1

    if k != int(theta.size):
        raise RuntimeError(
            f"ansatz parameter traversal consumed {k}, expected {int(theta.size)}"
        )
    return out


def _append_reference_state(circuit: QuantumCircuit, reference_state: np.ndarray) -> None:
    ref = np.asarray(reference_state, dtype=complex).reshape(-1)
    dim = int(1 << int(circuit.num_qubits))
    if ref.size != dim:
        raise ValueError(
            f"reference_state dimension {ref.size} does not match num_qubits={circuit.num_qubits}"
        )
    nrm = float(np.linalg.norm(ref))
    if nrm <= 0.0:
        raise ValueError("reference_state has zero norm")
    ref = ref / nrm

    nz = np.where(np.abs(ref) > 1e-12)[0]
    if nz.size == 1:
        idx = int(nz[0])
        phase = ref[idx]
        if abs(abs(phase) - 1.0) <= 1e-10:
            bit = format(idx, f"0{circuit.num_qubits}b")
            for q in range(circuit.num_qubits):
                if bit[circuit.num_qubits - 1 - q] == "1":
                    circuit.x(q)
            return

    circuit.initialize(ref, list(range(circuit.num_qubits)))


def _append_pauli_rotation_exyz(circuit: QuantumCircuit, *, label_exyz: str, angle: float) -> None:
    label = str(label_exyz).strip().lower()
    nq = int(circuit.num_qubits)
    if len(label) != nq:
        raise ValueError(f"Pauli label length mismatch: got {len(label)}, expected {nq}.")
    active: list[tuple[int, str]] = []
    for idx, ch in enumerate(label):
        if ch == "e":
            continue
        qubit = int(nq - 1 - idx)
        active.append((qubit, ch))
    if not active:
        return
    active.sort(key=lambda item: item[0])
    for qubit, ch in active:
        if ch == "x":
            circuit.h(qubit)
        elif ch == "y":
            circuit.sdg(qubit)
            circuit.h(qubit)
        elif ch == "z":
            pass
        else:
            raise ValueError(f"Unsupported Pauli letter {ch!r} in {label_exyz!r}.")
    active_qubits = [q for q, _ in active]
    if len(active_qubits) == 1:
        circuit.rz(float(angle), active_qubits[0])
    else:
        for control, target in zip(active_qubits[:-1], active_qubits[1:]):
            circuit.cx(control, target)
        circuit.rz(float(angle), active_qubits[-1])
        for control, target in reversed(list(zip(active_qubits[:-1], active_qubits[1:]))):
            circuit.cx(control, target)
    for qubit, ch in reversed(active):
        if ch == "x":
            circuit.h(qubit)
        elif ch == "y":
            circuit.h(qubit)
            circuit.s(qubit)


"""
U(theta_runtime) = Π_b Π_j exp(-i theta_bj c_bj P_bj), using serialized runtime layout order.
"""
def build_runtime_layout_circuit(
    layout: AnsatzParameterLayout,
    theta_runtime: np.ndarray | Sequence[float],
    num_qubits: int,
    *,
    reference_state: np.ndarray | None = None,
) -> QuantumCircuit:
    qc = QuantumCircuit(int(num_qubits))
    if reference_state is not None:
        _append_reference_state(qc, np.asarray(reference_state, dtype=complex).reshape(-1))
    theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if int(theta_arr.size) != int(layout.runtime_parameter_count):
        raise ValueError(
            f"theta_runtime length mismatch: got {theta_arr.size}, expected {layout.runtime_parameter_count}."
        )
    for block in layout.blocks:
        if int(block.runtime_count) <= 0:
            continue
        block_theta = theta_arr[int(block.runtime_start):int(block.runtime_stop)]
        for local_idx, spec in enumerate(block.terms):
            angle = 2.0 * float(block_theta[int(local_idx)]) * float(spec.coeff_real)
            _append_pauli_rotation_exyz(qc, label_exyz=str(spec.pauli_exyz), angle=float(angle))
    return qc


def pauli_poly_to_sparse_pauli_op(poly: Any, tol: float = 1e-12) -> SparsePauliOp:
    return _pauli_poly_to_sparse_pauli_op(poly, tol=tol)


def _ansatz_to_circuit(
    ansatz: Any,
    theta: np.ndarray,
    *,
    num_qubits: int,
    reference_state: np.ndarray | None = None,
    coefficient_tolerance: float = 1e-12,
) -> QuantumCircuit:
    """Convert existing hardcoded ansatz object into a Qiskit circuit."""
    qc = QuantumCircuit(int(num_qubits))
    if reference_state is not None:
        _append_reference_state(qc, np.asarray(reference_state, dtype=complex))

    terms = _ansatz_terms_with_parameters(ansatz, np.asarray(theta, dtype=float))
    synthesis = SuzukiTrotter(order=2, reps=1, preserve_order=True)

    for poly, angle in terms:
        qop = _pauli_poly_to_sparse_pauli_op(poly, tol=float(coefficient_tolerance))
        coeffs = np.asarray(qop.coeffs, dtype=complex).reshape(-1)
        if coeffs.size == 0 or np.max(np.abs(coeffs)) <= float(coefficient_tolerance):
            continue
        gate = PauliEvolutionGate(qop, time=float(angle), synthesis=synthesis)
        qc.append(gate, list(range(int(num_qubits))))
    return qc


def _load_fake_backend(name: str | None) -> tuple[Any, str]:
    class_name = str(name).strip() if name is not None else "FakeManilaV2"
    try:
        return _load_local_fake_backend_shared(class_name)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def list_local_fake_backend_names() -> tuple[str, ...]:
    names = _list_local_fake_backend_names_shared()
    if not names:
        raise RuntimeError(
            "Unable to import qiskit_ibm_runtime.fake_provider; install qiskit-ibm-runtime."
        )
    return tuple(names)


def compile_circuit_for_local_backend(
    circuit: QuantumCircuit,
    backend: Any,
    *,
    seed_transpiler: int,
    optimization_level: int = 1,
) -> dict[str, Any]:
    return _compile_circuit_for_backend_shared(
        circuit,
        backend,
        seed_transpiler=int(seed_transpiler),
        optimization_level=int(optimization_level),
    )


def _resolve_noise_backend(cfg: OracleConfig) -> tuple[Any, str, bool]:
    if bool(cfg.use_fake_backend):
        backend, name = _load_fake_backend(cfg.backend_name)
        return backend, name, True

    if cfg.backend_name is None:
        backend, name = _load_fake_backend("FakeManilaV2")
        return backend, name, True

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:
        raise RuntimeError(
            "qiskit_ibm_runtime is required for real backend lookup. "
            "Use --use-fake-backend or install/configure qiskit-ibm-runtime."
        ) from exc

    try:
        service = QiskitRuntimeService()
        backend = service.backend(str(cfg.backend_name))
        return backend, str(cfg.backend_name), False
    except Exception as exc:
        raise RuntimeError(
            f"Unable to resolve runtime backend '{cfg.backend_name}'. "
            "Check IBM Runtime credentials, backend name, or pass --use-fake-backend."
        ) from exc


def _compiled_metrics_payload(compiled: QuantumCircuit) -> dict[str, Any]:
    stats = dict(_compiled_gate_stats_shared(compiled))
    return {
        "compiled_depth": int(_safe_circuit_depth_shared(compiled)),
        "compiled_size": int(compiled.size()),
        "compiled_two_qubit_count": int(stats.get("compiled_count_2q", 0)),
        "compiled_cx_count": int(stats.get("compiled_cx_count", 0)),
        "compiled_ecr_count": int(stats.get("compiled_ecr_count", 0)),
        "compiled_op_counts": dict(stats.get("compiled_op_counts", {})),
    }


_OMP_SHM_MARKERS = (
    "OMP: Error #178",
    "Can't open SHM2",
    "Function Can't open SHM2 failed",
    "OMP: System error",
)
_AER_PREFLIGHT_OK_CACHE: set[tuple[str, int, int | None, bool, bool]] = set()
BACKEND_SCHEDULED_ATTRIBUTION_SLICES: tuple[str, ...] = (
    "readout_only",
    "gate_stateprep_only",
    "full",
)


def _tail_text(text: str, max_chars: int = 2400) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return "<no output captured>"
    if len(cleaned) <= int(max_chars):
        return cleaned
    return "..." + cleaned[-int(max_chars):]


def _looks_like_openmp_shm_abort(text: str) -> bool:
    lowered = str(text).lower()
    if not lowered:
        return False
    return any(str(marker).lower() in lowered for marker in _OMP_SHM_MARKERS)


def _apply_omp_env_workaround(cfg: OracleConfig) -> bool:
    if not bool(cfg.omp_shm_workaround):
        return False
    changed = False
    if os.environ.get("KMP_USE_SHM") != "0":
        os.environ["KMP_USE_SHM"] = "0"
        changed = True
    if os.environ.get("OMP_NUM_THREADS") != "1":
        os.environ["OMP_NUM_THREADS"] = "1"
        changed = True
    return changed


def _preflight_aer_environment(cfg: OracleConfig, mode: str) -> None:
    key = (
        str(mode),
        int(cfg.shots),
        (None if cfg.seed is None else int(cfg.seed)),
        bool(cfg.approximation),
        bool(cfg.abelian_grouping),
    )
    if key in _AER_PREFLIGHT_OK_CACHE:
        return

    payload = {
        "mode": str(mode),
        "shots": int(cfg.shots),
        "seed": (None if cfg.seed is None else int(cfg.seed)),
        "approximation": bool(cfg.approximation),
        "abelian_grouping": bool(cfg.abelian_grouping),
    }
    script = r"""
import json
import sys

cfg = json.loads(sys.argv[1])
mode = str(cfg.get("mode", "shots")).strip().lower()

from qiskit_aer.primitives import Estimator as AerEstimator

backend_options = {}
if mode == "aer_noise":
    from qiskit_aer.noise import NoiseModel
    backend_options["noise_model"] = NoiseModel()

run_options = {"shots": int(cfg["shots"])}
seed = cfg.get("seed", None)
if seed is not None:
    run_options["seed"] = int(seed)
    run_options["seed_simulator"] = int(seed)

_ = AerEstimator(
    backend_options=backend_options if backend_options else None,
    run_options=run_options,
    approximation=bool(cfg.get("approximation", False)),
    abelian_grouping=bool(cfg.get("abelian_grouping", True)),
)
print("AER_PREFLIGHT_OK")
"""
    env = None
    if bool(cfg.omp_shm_workaround):
        env = dict(os.environ)
        env["KMP_USE_SHM"] = "0"
        env["OMP_NUM_THREADS"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(payload, sort_keys=True)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if int(result.returncode) != 0:
        combined = f"{result.stdout}\n{result.stderr}"
        detail_tail = _tail_text(combined)
        if _looks_like_openmp_shm_abort(combined):
            raise RuntimeError(
                "Aer preflight failed due to OpenMP shared-memory restrictions in this environment "
                "(detected OMP/SHM2 failure). This is an environment-level crash path, not a script logic "
                "error. Modes 'shots' and 'aer_noise' are local/offline and do not require IBM Runtime "
                "credentials. Run this command in a shell/runtime with working shared-memory support "
                "(for example, a non-sandbox terminal with functional /dev/shm or equivalent). "
                f"Preflight stderr/stdout tail:\n{detail_tail}"
            )
        raise RuntimeError(
            "Aer preflight failed before noisy execution started. "
            f"Preflight stderr/stdout tail:\n{detail_tail}"
        )

    _AER_PREFLIGHT_OK_CACHE.add(key)


def _build_estimator(
    cfg: OracleConfig,
) -> tuple[Any, Any | None, NoiseBackendInfo, Any | None]:
    mode = str(cfg.noise_mode).strip().lower()
    mitigation_cfg = normalize_mitigation_config(getattr(cfg, "mitigation", "none"))
    symmetry_cfg = normalize_symmetry_mitigation_config(getattr(cfg, "symmetry_mitigation", "off"))
    if mode not in {"ideal", "shots", "aer_noise", "runtime"}:
        raise ValueError(f"Unsupported noise_mode: {mode}")

    if mode == "ideal":
        try:
            from qiskit.primitives import StatevectorEstimator
        except Exception as exc:
            raise RuntimeError(
                "Failed to import StatevectorEstimator. Ensure qiskit primitives are available."
            ) from exc
        estimator = StatevectorEstimator()
        info = NoiseBackendInfo(
            noise_mode=mode,
            estimator_kind="qiskit.primitives.StatevectorEstimator",
            backend_name="statevector_simulator",
            using_fake_backend=False,
            details={
                "shots": None,
                "mitigation": dict(mitigation_cfg),
                "symmetry_mitigation": dict(symmetry_cfg),
            },
        )
        return estimator, None, info, None

    if mode in {"shots", "aer_noise"}:
        if str(os.environ.get("HH_FORCE_SAMPLER_FALLBACK", "0")).strip() == "1":
            raise RuntimeError(
                "OMP: Error #178: Forced sampler fallback via HH_FORCE_SAMPLER_FALLBACK=1."
            )
        env_workaround_applied = _apply_omp_env_workaround(cfg)
        _preflight_aer_environment(cfg, mode)
        try:
            from qiskit_aer.primitives import Estimator as AerEstimator
        except Exception as exc:
            raise RuntimeError(
                "Failed to import qiskit_aer.primitives.Estimator. Install qiskit-aer."
            ) from exc

        backend_options: dict[str, Any] = {}
        backend_name = "aer_simulator"
        using_fake = False
        details: dict[str, Any] = {
            "shots": int(cfg.shots),
            "aer_failed": False,
            "fallback_used": False,
            "fallback_mode": str(cfg.aer_fallback_mode),
            "fallback_reason": "",
            "env_workaround_applied": bool(cfg.omp_shm_workaround or env_workaround_applied),
            "mitigation": dict(mitigation_cfg),
            "symmetry_mitigation": dict(symmetry_cfg),
        }

        if mode == "aer_noise":
            try:
                from qiskit_aer.noise import NoiseModel
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import qiskit_aer.noise.NoiseModel for aer_noise mode."
                ) from exc
            backend_obj, backend_name, using_fake = _resolve_noise_backend(cfg)
            noise_model = NoiseModel.from_backend(backend_obj)
            backend_options["noise_model"] = noise_model
            details["noise_model_basis_gates"] = list(getattr(noise_model, "basis_gates", []))

        run_options: dict[str, Any] = {"shots": int(cfg.shots)}
        if cfg.seed is not None:
            run_options["seed"] = int(cfg.seed)
            run_options["seed_simulator"] = int(cfg.seed)
        estimator = AerEstimator(
            backend_options=backend_options if backend_options else None,
            run_options=run_options,
            approximation=bool(cfg.approximation),
            abelian_grouping=bool(cfg.abelian_grouping),
        )
        info = NoiseBackendInfo(
            noise_mode=mode,
            estimator_kind="qiskit_aer.primitives.Estimator",
            backend_name=backend_name,
            using_fake_backend=using_fake,
            details=details,
        )
        return estimator, None, info, None

    # mode == "runtime"
    try:
        from qiskit_ibm_runtime import (
            QiskitRuntimeService,
            Session,
            EstimatorV2 as RuntimeEstimatorV2,
        )
    except Exception as exc:
        raise RuntimeError(
            "runtime mode requires qiskit-ibm-runtime. Install and configure IBM Runtime."
        ) from exc

    if cfg.backend_name is None:
        raise RuntimeError(
            "runtime mode requires --backend-name <ibm_backend>."
        )

    runtime_profile_cfg = normalize_runtime_estimator_profile_config(
        getattr(cfg, "runtime_profile", "legacy_runtime_v0")
    )
    runtime_session_cfg = normalize_runtime_session_policy_config(
        getattr(cfg, "runtime_session", "prefer_session")
    )
    try:
        service = QiskitRuntimeService()
        backend = service.backend(str(cfg.backend_name))
        session = None
        runtime_mode = backend
        runtime_mode_kind = "backend"
        session_init_error = None
        session_policy = str(runtime_session_cfg.get("mode", "prefer_session"))
        if session_policy != "backend_only":
            try:
                try:
                    session = Session(service=service, backend=backend)
                except TypeError:
                    session = Session(backend=backend)
                runtime_mode = session
                runtime_mode_kind = "session"
            except Exception as session_exc:
                session = None
                session_init_error = f"{type(session_exc).__name__}: {session_exc}"
                if session_policy == "require_session":
                    raise RuntimeError(
                        "Runtime session initialization failed while session batching was required. "
                        f"Details: {session_init_error}"
                    ) from session_exc
        else:
            session_init_error = "backend_only_requested"
        estimator = RuntimeEstimatorV2(mode=runtime_mode)
        runtime_mitigation_details = _configure_runtime_estimator_options(
            estimator,
            cfg=cfg,
            mitigation_cfg=mitigation_cfg,
            runtime_profile_cfg=runtime_profile_cfg,
        )
        info = NoiseBackendInfo(
            noise_mode=mode,
            estimator_kind="qiskit_ibm_runtime.EstimatorV2",
            backend_name=str(cfg.backend_name),
            using_fake_backend=False,
            details={
                "shots": int(cfg.shots),
                "runtime_execution_mode": str(runtime_mode_kind),
                "runtime_session_policy": dict(runtime_session_cfg),
                "runtime_profile": dict(runtime_profile_cfg),
                "session_fallback_reason": session_init_error,
                "mitigation": dict(mitigation_cfg),
                "runtime_mitigation": dict(runtime_mitigation_details),
                "symmetry_mitigation": dict(symmetry_cfg),
            },
        )
        return estimator, session, info, backend
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize IBM Runtime Estimator. "
            "Verify IBM credentials (`QISKIT_IBM_TOKEN`), backend availability, account access, and session policy."
        ) from exc


def _configure_runtime_estimator_options(
    estimator: Any,
    *,
    cfg: OracleConfig,
    mitigation_cfg: Mapping[str, Any],
    runtime_profile_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    mitigation_mode = str(mitigation_cfg.get("mode", "none")).strip().lower() or "none"
    profile_name = str(runtime_profile_cfg.get("name", "legacy_runtime_v0"))
    runtime_details: dict[str, Any] = {
        "mode": str(mitigation_mode),
        "engine": "none",
        "requested_local_readout_strategy": mitigation_cfg.get("local_readout_strategy", None),
        "profile_name": str(profile_name),
        "applied": False,
        "explicit_profile": bool(profile_name != "legacy_runtime_v0"),
        "requested_profile": dict(runtime_profile_cfg),
    }
    options = getattr(estimator, "options", None)
    if options is None:
        return runtime_details

    def _require_path(root: Any, *attrs: str) -> Any:
        cur = root
        traversed: list[str] = []
        for attr in attrs:
            traversed.append(str(attr))
            if not hasattr(cur, attr):
                raise RuntimeError(
                    "Installed qiskit-ibm-runtime does not support requested Runtime option path "
                    f"{'.'.join(traversed)!r} for profile {profile_name!r}."
                )
            cur = getattr(cur, attr)
        return cur

    def _maybe_set(attr_root: Any, attr_name: str, value: Any, *, required: bool) -> None:
        if value is None:
            return
        if required:
            _require_path(attr_root, attr_name)
        if hasattr(attr_root, attr_name):
            setattr(attr_root, attr_name, value)

    try:
        if hasattr(options, "default_shots"):
            options.default_shots = int(
                cfg.shots if runtime_profile_cfg.get("default_shots", None) is None else runtime_profile_cfg["default_shots"]
            )
        if runtime_profile_cfg.get("default_precision", None) is not None and hasattr(options, "default_precision"):
            options.default_precision = float(runtime_profile_cfg["default_precision"])
        if runtime_profile_cfg.get("max_execution_time", None) is not None and hasattr(options, "max_execution_time"):
            options.max_execution_time = int(runtime_profile_cfg["max_execution_time"])
        if cfg.seed is not None and hasattr(options, "seed_estimator"):
            options.seed_estimator = int(cfg.seed)
        if runtime_profile_cfg.get("init_qubits", None) is not None:
            _maybe_set(getattr(options, "execution", None), "init_qubits", bool(runtime_profile_cfg["init_qubits"]), required=bool(profile_name != "legacy_runtime_v0"))
        if runtime_profile_cfg.get("resilience_level", None) is not None:
            _maybe_set(options, "resilience_level", int(runtime_profile_cfg["resilience_level"]), required=bool(profile_name != "legacy_runtime_v0"))

        explicit_profile = bool(profile_name != "legacy_runtime_v0")
        if explicit_profile:
            _maybe_set(getattr(options, "resilience", None), "measure_mitigation", bool(runtime_profile_cfg.get("measure_mitigation", False)), required=True)
            _maybe_set(getattr(options, "resilience", None), "zne_mitigation", bool(runtime_profile_cfg.get("zne_mitigation", False)), required=True)
            _maybe_set(getattr(options, "resilience", None), "pec_mitigation", bool(runtime_profile_cfg.get("pec_mitigation", False)), required=True)
            _maybe_set(getattr(options, "twirling", None), "enable_measure", bool(runtime_profile_cfg.get("measure_twirling", False)), required=True)
            _maybe_set(getattr(options, "twirling", None), "enable_gates", bool(runtime_profile_cfg.get("gate_twirling", False)), required=True)
            _maybe_set(getattr(options, "twirling", None), "strategy", runtime_profile_cfg.get("twirling_strategy", None), required=True)
            _maybe_set(getattr(options, "dynamical_decoupling", None), "enable", bool(runtime_profile_cfg.get("dd_enable", False)), required=True)
            if runtime_profile_cfg.get("dd_enable", False):
                _maybe_set(getattr(options, "dynamical_decoupling", None), "sequence_type", runtime_profile_cfg.get("dd_sequence", None), required=True)
            if runtime_profile_cfg.get("zne_mitigation", False):
                zne_obj = _require_path(options, "resilience", "zne")
                if runtime_profile_cfg.get("zne_noise_factors", None):
                    _maybe_set(zne_obj, "noise_factors", tuple(float(x) for x in runtime_profile_cfg.get("zne_noise_factors", [])), required=True)
                if runtime_profile_cfg.get("zne_extrapolator", None):
                    _maybe_set(zne_obj, "extrapolator", tuple(str(x) for x in runtime_profile_cfg.get("zne_extrapolator", [])), required=True)
            runtime_details.update(
                {
                    "engine": "runtime_profile",
                    "provider_strategy": "explicit_profile",
                    "applied": True,
                    "measure_mitigation": bool(runtime_profile_cfg.get("measure_mitigation", False)),
                    "measure_twirling": bool(runtime_profile_cfg.get("measure_twirling", False)),
                    "gate_twirling": bool(runtime_profile_cfg.get("gate_twirling", False)),
                    "twirling_strategy": runtime_profile_cfg.get("twirling_strategy", None),
                    "zne_mitigation": bool(runtime_profile_cfg.get("zne_mitigation", False)),
                    "zne_noise_factors": [float(x) for x in runtime_profile_cfg.get("zne_noise_factors", [])],
                    "zne_extrapolator": [str(x) for x in runtime_profile_cfg.get("zne_extrapolator", [])],
                    "dd_enable": bool(runtime_profile_cfg.get("dd_enable", False)),
                    "dd_sequence": runtime_profile_cfg.get("dd_sequence", None),
                }
            )
            return runtime_details

        if hasattr(options, "resilience") and hasattr(options.resilience, "measure_mitigation"):
            options.resilience.measure_mitigation = bool(mitigation_mode == "readout")
        if hasattr(options, "resilience") and hasattr(options.resilience, "zne_mitigation"):
            options.resilience.zne_mitigation = bool(mitigation_mode == "zne")
        if (
            mitigation_mode == "zne"
            and hasattr(options, "resilience")
            and hasattr(options.resilience, "zne")
            and mitigation_cfg.get("zne_scales")
        ):
            options.resilience.zne.noise_factors = tuple(
                float(x) for x in mitigation_cfg.get("zne_scales", [])
            )
        if hasattr(options, "dynamical_decoupling") and hasattr(options.dynamical_decoupling, "enable"):
            options.dynamical_decoupling.enable = bool(mitigation_mode == "dd")
        if (
            mitigation_mode == "dd"
            and mitigation_cfg.get("dd_sequence") is not None
            and hasattr(options, "dynamical_decoupling")
            and hasattr(options.dynamical_decoupling, "sequence_type")
        ):
            options.dynamical_decoupling.sequence_type = str(mitigation_cfg.get("dd_sequence"))
    except Exception:
        if profile_name != "legacy_runtime_v0":
            raise
        return runtime_details

    if mitigation_mode == "readout":
        runtime_details.update(
            {
                "engine": "runtime_resilience.measure_mitigation",
                "provider_strategy": "builtin_measure_mitigation",
                "applied": True,
            }
        )
    elif mitigation_mode == "zne":
        runtime_details.update(
            {
                "engine": "runtime_resilience.zne_mitigation",
                "noise_factors": [float(x) for x in mitigation_cfg.get("zne_scales", [])],
                "applied": True,
            }
        )
    elif mitigation_mode == "dd":
        runtime_details.update(
            {
                "engine": "runtime_dynamical_decoupling",
                "sequence_type": (
                    None if mitigation_cfg.get("dd_sequence") is None else str(mitigation_cfg.get("dd_sequence"))
                ),
                "applied": True,
            }
        )
    return runtime_details


def _configure_runtime_sampler_options(
    sampler: Any,
    *,
    cfg: OracleConfig,
    runtime_profile_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    profile_name = str(runtime_profile_cfg.get("name", "legacy_runtime_v0"))
    runtime_details: dict[str, Any] = {
        "profile_name": str(profile_name),
        "applied": False,
        "requested_profile": dict(runtime_profile_cfg),
    }
    options = getattr(sampler, "options", None)
    if options is None:
        return runtime_details

    def _maybe_set(attr_root: Any, attr_name: str, value: Any) -> None:
        if value is None or attr_root is None or not hasattr(attr_root, attr_name):
            return
        setattr(attr_root, attr_name, value)

    try:
        if hasattr(options, "default_shots"):
            options.default_shots = int(
                cfg.shots
                if runtime_profile_cfg.get("default_shots", None) is None
                else runtime_profile_cfg["default_shots"]
            )
        if runtime_profile_cfg.get("max_execution_time", None) is not None and hasattr(
            options, "max_execution_time"
        ):
            options.max_execution_time = int(runtime_profile_cfg["max_execution_time"])
        _maybe_set(
            getattr(options, "execution", None),
            "init_qubits",
            runtime_profile_cfg.get("init_qubits", None),
        )
        explicit_profile = bool(profile_name != "legacy_runtime_v0")
        if explicit_profile:
            _maybe_set(
                getattr(options, "twirling", None),
                "enable_measure",
                bool(runtime_profile_cfg.get("measure_twirling", False)),
            )
            _maybe_set(
                getattr(options, "twirling", None),
                "enable_gates",
                bool(runtime_profile_cfg.get("gate_twirling", False)),
            )
            _maybe_set(
                getattr(options, "twirling", None),
                "strategy",
                runtime_profile_cfg.get("twirling_strategy", None),
            )
            _maybe_set(
                getattr(options, "dynamical_decoupling", None),
                "enable",
                bool(runtime_profile_cfg.get("dd_enable", False)),
            )
            if runtime_profile_cfg.get("dd_enable", False):
                _maybe_set(
                    getattr(options, "dynamical_decoupling", None),
                    "sequence_type",
                    runtime_profile_cfg.get("dd_sequence", None),
                )
        runtime_details.update(
            {
                "applied": True,
                "measure_twirling": bool(runtime_profile_cfg.get("measure_twirling", False)),
                "gate_twirling": bool(runtime_profile_cfg.get("gate_twirling", False)),
                "twirling_strategy": runtime_profile_cfg.get("twirling_strategy", None),
                "dd_enable": bool(runtime_profile_cfg.get("dd_enable", False)),
                "dd_sequence": runtime_profile_cfg.get("dd_sequence", None),
            }
        )
    except Exception as exc:
        runtime_details["applied"] = False
        runtime_details["error"] = f"{type(exc).__name__}: {exc}"
    return runtime_details


def _extract_expectation_value(result: Any) -> float:
    if hasattr(result, "values"):
        vals = np.asarray(getattr(result, "values"), dtype=float).reshape(-1)
        if vals.size > 0:
            return float(vals[0])

    try:
        first = result[0]
    except Exception:
        first = None

    if first is not None:
        data = getattr(first, "data", None)
        if data is not None and hasattr(data, "evs"):
            evs = np.asarray(getattr(data, "evs"), dtype=float).reshape(-1)
            if evs.size > 0:
                return float(evs[0])
        if hasattr(first, "value"):
            return float(np.real(getattr(first, "value")))
        if hasattr(first, "evs"):
            evs = np.asarray(getattr(first, "evs"), dtype=float).reshape(-1)
            if evs.size > 0:
                return float(evs[0])

    if hasattr(result, "evs"):
        evs = np.asarray(getattr(result, "evs"), dtype=float).reshape(-1)
        if evs.size > 0:
            return float(evs[0])

    raise RuntimeError(
        f"Unable to extract expectation value from estimator result type: {type(result)!r}"
    )


def _extract_sampler_counts(result: Any) -> dict[str, int]:
    try:
        first = result[0]
    except Exception as exc:
        raise RuntimeError(
            f"Unable to access first sampler pub result from object {type(result)!r}."
        ) from exc

    if hasattr(first, "join_data"):
        joined = first.join_data()
        if hasattr(joined, "get_counts"):
            counts = joined.get_counts()
            return {str(bitstr): int(ct) for bitstr, ct in dict(counts).items()}

    data = getattr(first, "data", None)
    if data is not None:
        for attr in ("c", "m", "meas", "cr"):
            if hasattr(data, attr):
                bit_array = getattr(data, attr)
                if hasattr(bit_array, "get_counts"):
                    counts = bit_array.get_counts()
                    return {str(bitstr): int(ct) for bitstr, ct in dict(counts).items()}

    raise RuntimeError(
        f"Unable to extract counts from sampler result type: {type(result)!r}"
    )


def _datetime_to_utc_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return str(value)
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        return str(value)
    except Exception:
        return None


def _runtime_job_record_from_job(
    job: Any,
    *,
    repeat_index: int,
    call_path: str,
    expectation_value: float | None = None,
    error: str | None = None,
) -> RuntimeJobRecord:
    status_val = None
    try:
        status_attr = getattr(job, "status", None)
        status_val = status_attr() if callable(status_attr) else status_attr
    except Exception as exc:  # pragma: no cover - defensive only
        status_val = f"STATUS_ERROR:{type(exc).__name__}"

    metrics_payload: Mapping[str, Any] = {}
    try:
        metrics_attr = getattr(job, "metrics", None)
        metrics_val = metrics_attr() if callable(metrics_attr) else metrics_attr
        if isinstance(metrics_val, Mapping):
            metrics_payload = metrics_val
    except Exception:
        metrics_payload = {}
    timestamps = metrics_payload.get("timestamps", {}) if isinstance(metrics_payload, Mapping) else {}
    usage_payload = metrics_payload.get("usage", {}) if isinstance(metrics_payload, Mapping) else {}
    if not isinstance(usage_payload, Mapping):
        usage_payload = {}

    try:
        job_id_attr = getattr(job, "job_id", None)
        job_id = job_id_attr() if callable(job_id_attr) else job_id_attr
    except Exception:
        job_id = None
    try:
        backend_attr = getattr(job, "backend", None)
        backend_obj = backend_attr() if callable(backend_attr) else backend_attr
        backend_name = getattr(backend_obj, "name", None)
        if callable(backend_name):
            backend_name = backend_name()
        if backend_name is None:
            backend_name = getattr(backend_obj, "_instance", None)
    except Exception:
        backend_name = None
    try:
        session_attr = getattr(job, "session_id", None)
        session_id = session_attr() if callable(session_attr) else session_attr
    except Exception:
        session_id = None
    try:
        usage_attr = getattr(job, "usage", None)
        usage_direct = usage_attr() if callable(usage_attr) else usage_attr
    except Exception:
        usage_direct = None
    usage_quantum_seconds = usage_payload.get("quantum_seconds", usage_direct)
    try:
        usage_quantum_seconds = (
            None if usage_quantum_seconds is None else float(usage_quantum_seconds)
        )
    except Exception:
        usage_quantum_seconds = None

    return RuntimeJobRecord(
        job_id=(None if job_id in {None, ""} else str(job_id)),
        repeat_index=int(repeat_index),
        call_path=str(call_path),
        status=("unknown" if status_val in {None, ""} else str(status_val)),
        created_utc=_datetime_to_utc_str(
            timestamps.get("created", None)
            if isinstance(timestamps, Mapping)
            else getattr(job, "creation_date", None)
        ),
        running_utc=_datetime_to_utc_str(
            timestamps.get("running", None) if isinstance(timestamps, Mapping) else None
        ),
        completed_utc=_datetime_to_utc_str(
            timestamps.get("finished", None) if isinstance(timestamps, Mapping) else None
        ),
        expectation_value=(
            None if expectation_value is None else float(np.real(expectation_value))
        ),
        error=(None if error in {None, ""} else str(error)),
        backend_name=(None if backend_name in {None, ""} else str(backend_name)),
        session_id=(None if session_id in {None, ""} else str(session_id)),
        usage_quantum_seconds=usage_quantum_seconds,
    )


def _emit_runtime_job_event(
    observer: Callable[[dict[str, Any]], None] | None,
    *,
    event: str,
    record: RuntimeJobRecord,
    context: Mapping[str, Any] | None = None,
) -> None:
    if observer is None:
        return
    payload = {"event": str(event), "job": asdict(record)}
    if context is not None:
        payload.update({str(k): v for k, v in dict(context).items()})
    observer(payload)


def _oracle_estimate_from_samples(values: Sequence[float], *, aggregate: str) -> OracleEstimate:
    arr = np.asarray(list(values), dtype=float)
    stdev = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    stderr = float(stdev / np.sqrt(float(arr.size))) if arr.size > 0 else float("nan")
    if str(aggregate).strip().lower() == "median":
        agg = float(np.median(arr))
    else:
        agg = float(np.mean(arr))
    return OracleEstimate(
        mean=agg,
        std=stdev,
        stdev=stdev,
        stderr=stderr,
        n_samples=int(arr.size),
        raw_values=[float(x) for x in arr.tolist()],
        aggregate=str(aggregate).strip().lower(),
    )


def _qwc_compatible(lhs: str, rhs: str) -> bool:
    left = str(lhs).upper()
    right = str(rhs).upper()
    if len(left) != len(right):
        return False
    for lch, rch in zip(left, right):
        if lch == "I" or rch == "I" or lch == rch:
            continue
        return False
    return True


def _merge_qwc_basis(lhs: str, rhs: str) -> str:
    left = str(lhs).upper()
    right = str(rhs).upper()
    if len(left) != len(right):
        raise ValueError("Cannot merge QWC bases of different lengths.")
    merged: list[str] = []
    for lch, rch in zip(left, right):
        if lch == "I":
            merged.append(rch)
        elif rch == "I" or lch == rch:
            merged.append(lch)
        else:
            raise ValueError(f"Non-QWC merge attempted for {lhs!r} and {rhs!r}.")
    return "".join(merged)


def _group_observable_terms_by_qwc_basis(
    terms: Sequence[tuple[str, complex]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for label, coeff in terms:
        label_s = str(label).upper()
        placed = False
        for group in groups:
            if _qwc_compatible(str(group["basis_label"]), label_s):
                group["basis_label"] = _merge_qwc_basis(str(group["basis_label"]), label_s)
                group["terms"].append((label_s, complex(coeff)))
                placed = True
                break
        if not placed:
            groups.append(
                {
                    "basis_label": str(label_s),
                    "terms": [(str(label_s), complex(coeff))],
                }
            )
    return groups


def _subset_parity_from_counts(counts: Mapping[str, int], classical_indices: Sequence[int]) -> float:
    indices = [int(idx) for idx in classical_indices]
    if not indices:
        return 1.0
    shots = int(sum(int(v) for v in counts.values()))
    if shots <= 0:
        raise RuntimeError("Sampler returned zero total shots.")
    acc = 0.0
    width = max(indices) + 1
    for bitstr_raw, ct in counts.items():
        bitstr = str(bitstr_raw).replace(" ", "")
        if len(bitstr) < width:
            bitstr = bitstr.zfill(width)
        ones = sum(1 for idx in indices if bitstr[-1 - int(idx)] == "1")
        parity = -1.0 if (ones % 2) else 1.0
        acc += float(parity) * float(ct)
    return float(acc / float(shots))


def _weighted_mean_stderr(values: Sequence[float], weights: Sequence[float]) -> tuple[float, float]:
    vals = np.asarray([float(v) for v in values], dtype=float)
    wts = np.asarray([max(1.0, float(w)) for w in weights], dtype=float)
    if vals.size <= 0:
        return 0.0, float("nan")
    mean = float(np.average(vals, weights=wts))
    if vals.size <= 1:
        return float(mean), 0.0
    var = float(np.average((vals - mean) ** 2, weights=wts))
    stdev = float(np.sqrt(max(var, 0.0)))
    n_eff = float((wts.sum() ** 2) / max(np.sum(wts**2), 1.0))
    stderr = float(stdev / np.sqrt(max(n_eff, 1.0)))
    return float(mean), float(stderr)


def _record_term_expectation(
    record: RawMeasurementRecord,
    term_label_ixyz: str,
) -> float:
    label = str(term_label_ixyz).upper()
    measured_logical = tuple(int(q) for q in record.measured_logical_qubits)
    n = int(len(label))
    subset_positions = [
        int(idx)
        for idx, logical_q in enumerate(measured_logical)
        if label[n - 1 - int(logical_q)] != "I"
    ]
    return float(_subset_parity_from_counts(record.counts, subset_positions))


def _reduce_grouped_counts_to_observable(
    observable: SparsePauliOp,
    records: Sequence[RawMeasurementRecord],
    *,
    aggregate: str,
    expected_repeat_count: int | None = None,
) -> RawObservableEstimate:
    observable_terms = [(str(label).upper(), complex(coeff)) for label, coeff in observable.to_list()]
    identity_total = 0.0 + 0.0j
    non_identity_terms: list[tuple[str, complex]] = []
    for label, coeff in observable_terms:
        if all(ch == "I" for ch in label):
            identity_total += complex(coeff)
        else:
            non_identity_terms.append((label, coeff))

    if abs(float(np.imag(identity_total))) > 1e-9:
        raise RuntimeError("Observable identity contribution has non-negligible imaginary component.")

    term_count = int(len(observable_terms))
    group_count = int(len(_group_observable_terms_by_qwc_basis(non_identity_terms)))
    record_count = int(len(records))
    total_shots = int(sum(int(rec.shots_completed) for rec in records))
    normalized_aggregate = str(aggregate).strip().lower()

    if not non_identity_terms:
        return RawObservableEstimate(
            mean=float(np.real(identity_total)),
            std=0.0,
            stdev=0.0,
            stderr=0.0,
            n_samples=1,
            raw_values=(float(np.real(identity_total)),),
            aggregate=normalized_aggregate,
            total_shots=0,
            group_count=0,
            term_count=term_count,
            record_count=0,
            reduction_mode="repeat_aligned_full_observable",
        )

    groups = _group_observable_terms_by_qwc_basis(non_identity_terms)
    records_by_group_repeat: dict[tuple[str, int], list[RawMeasurementRecord]] = {}
    repeat_indices: set[int] = set()
    for rec in records:
        key = (str(rec.basis_label).upper(), int(rec.repeat_index))
        records_by_group_repeat.setdefault(key, []).append(rec)
        repeat_indices.add(int(rec.repeat_index))

    repeat_aligned = bool(records)
    if repeat_aligned and expected_repeat_count is not None:
        expected_indices = set(range(int(expected_repeat_count)))
        if repeat_indices != expected_indices:
            repeat_aligned = False
    if repeat_aligned:
        for group in groups:
            basis = str(group["basis_label"]).upper()
            for repeat_idx in sorted(repeat_indices):
                if len(records_by_group_repeat.get((basis, int(repeat_idx)), [])) != 1:
                    repeat_aligned = False
                    break
            if not repeat_aligned:
                break

    if repeat_aligned:
        values: list[float] = []
        for repeat_idx in sorted(repeat_indices):
            total = complex(identity_total)
            for group in groups:
                basis = str(group["basis_label"]).upper()
                rec = records_by_group_repeat[(basis, int(repeat_idx))][0]
                for term_label, coeff in group["terms"]:
                    total += complex(coeff) * complex(_record_term_expectation(rec, str(term_label)), 0.0)
            if abs(float(np.imag(total))) > 1e-9:
                raise RuntimeError("Reduced observable has non-negligible imaginary component.")
            values.append(float(np.real(total)))
        estimate = _oracle_estimate_from_samples(values, aggregate=normalized_aggregate)
        return RawObservableEstimate(
            mean=float(estimate.mean),
            std=float(estimate.std),
            stdev=float(estimate.stdev),
            stderr=float(estimate.stderr),
            n_samples=int(estimate.n_samples),
            raw_values=tuple(float(x) for x in estimate.raw_values),
            aggregate=str(estimate.aggregate),
            total_shots=int(total_shots),
            group_count=group_count,
            term_count=term_count,
            record_count=record_count,
            reduction_mode="repeat_aligned_full_observable",
        )

    total_mean = float(np.real(identity_total))
    total_stderr_sq = 0.0
    records_by_basis: dict[str, list[RawMeasurementRecord]] = {}
    for rec in records:
        records_by_basis.setdefault(str(rec.basis_label).upper(), []).append(rec)
    for group in groups:
        basis = str(group["basis_label"]).upper()
        group_records = list(records_by_basis.get(basis, []))
        if not group_records:
            raise RuntimeError(f"Missing raw records for grouped basis {basis!r}.")
        for term_label, coeff in group["terms"]:
            values = [_record_term_expectation(rec, str(term_label)) for rec in group_records]
            weights = [float(max(1, int(rec.shots_completed))) for rec in group_records]
            mean_term, stderr_term = _weighted_mean_stderr(values, weights)
            coeff_c = complex(coeff)
            if abs(float(np.imag(coeff_c))) > 1e-9:
                raise RuntimeError(f"Observable term {term_label!r} has non-negligible imaginary coefficient.")
            total_mean += float(np.real(coeff_c)) * float(mean_term)
            total_stderr_sq += (abs(float(np.real(coeff_c))) * float(stderr_term)) ** 2
    total_stderr = float(np.sqrt(max(total_stderr_sq, 0.0)))
    return RawObservableEstimate(
        mean=float(total_mean),
        std=float(total_stderr),
        stdev=float(total_stderr),
        stderr=float(total_stderr),
        n_samples=0,
        raw_values=tuple(),
        aggregate="weighted_term_fallback",
        total_shots=int(total_shots),
        group_count=group_count,
        term_count=term_count,
        record_count=record_count,
        reduction_mode="weighted_term_fallback",
    )


def _write_raw_measurement_record(path: str, record: RawMeasurementRecord) -> None:
    path_s = str(path).strip()
    if path_s == "":
        raise ValueError("raw artifact path must be non-empty.")
    os.makedirs(os.path.dirname(path_s) or ".", exist_ok=True)
    line = _stable_json_dumps(asdict(record)) + "\n"
    if path_s.endswith(".gz"):
        with gzip.open(path_s, "at", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
        return
    with open(path_s, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()


def _runtime_backend_supports_direct_run(backend: Any) -> bool:
    if backend is None:
        return False
    module_name = str(getattr(type(backend), "__module__", "")).lower()
    class_name = str(getattr(type(backend), "__name__", "")).lower()
    if "qiskit_ibm_runtime" in module_name or class_name == "ibmbackend":
        return False
    run_attr = getattr(backend, "run", None)
    return callable(run_attr)


def _resolve_raw_transport(cfg: OracleConfig, backend_target: Any | None) -> str:
    requested = str(getattr(cfg, "raw_transport", "auto")).strip().lower() or "auto"
    if requested not in {"auto", "sampler_v2", "backend_run"}:
        raise ValueError(f"Unsupported raw transport {requested!r}.")
    mode = str(cfg.noise_mode).strip().lower()
    if mode == "backend_scheduled":
        if requested == "sampler_v2":
            raise ValueError("backend_scheduled raw execution does not support sampler_v2 transport.")
        return "backend_run"
    if mode != "runtime":
        raise ValueError(f"Raw execution unsupported for noise mode {mode!r}.")
    if requested == "auto":
        return "sampler_v2"
    if requested == "backend_run" and not _runtime_backend_supports_direct_run(backend_target):
        raise ValueError(
            "raw transport 'backend_run' is not supported for the resolved runtime backend; "
            "use raw_transport='sampler_v2' on current IBM Runtime backends."
        )
    return str(requested)


def _run_raw_transport_job(
    *,
    transport: str,
    sampler: Any | None,
    backend_target: Any | None,
    circuit: QuantumCircuit,
    shots: int,
    repeat_index: int,
    seed: int,
    job_observer: Callable[[dict[str, Any]], None] | None = None,
    job_context: Mapping[str, Any] | None = None,
) -> SamplerExecutionResult:
    transport_key = str(transport).strip().lower()
    if transport_key == "sampler_v2":
        if sampler is None:
            raise RuntimeError("sampler_v2 raw transport requires an initialized sampler.")
        return _run_sampler_job(
            sampler,
            circuit,
            shots=int(shots),
            repeat_index=int(repeat_index),
            job_observer=job_observer,
            job_context=job_context,
        )
    if transport_key != "backend_run":
        raise ValueError(f"Unsupported raw transport {transport!r}.")
    if backend_target is None or not callable(getattr(backend_target, "run", None)):
        raise RuntimeError("backend_run raw transport requires a direct-run backend target.")
    job = None
    try:
        kwargs: dict[str, Any] = {"shots": int(shots)}
        if hasattr(backend_target, "options") or "fake" in str(type(backend_target)).lower():
            kwargs["seed_simulator"] = int(seed) + int(repeat_index)
        job = backend_target.run(circuit, **kwargs)
        submitted = _runtime_job_record_from_job(
            job,
            repeat_index=int(repeat_index),
            call_path="backend_run_counts",
        )
        _emit_runtime_job_event(
            job_observer,
            event="submitted",
            record=submitted,
            context=job_context,
        )
        result = job.result()
        counts = result.get_counts()
        if isinstance(counts, list):
            counts = counts[0]
        completed = _runtime_job_record_from_job(
            job,
            repeat_index=int(repeat_index),
            call_path="backend_run_counts",
        )
        _emit_runtime_job_event(
            job_observer,
            event="completed",
            record=completed,
            context=job_context,
        )
        return SamplerExecutionResult(
            counts={str(bitstr): int(ct) for bitstr, ct in dict(counts).items()},
            job_records=(completed,),
            used_call_path="backend_run_counts",
        )
    except Exception as exc:
        if job is not None:
            failed = _runtime_job_record_from_job(
                job,
                repeat_index=int(repeat_index),
                call_path="backend_run_counts",
                error=f"{type(exc).__name__}: {exc}",
            )
            _emit_runtime_job_event(
                job_observer,
                event="failed",
                record=failed,
                context=job_context,
            )
            raise SubmittedRuntimeJobError(
                "Raw backend.run execution failed after a job was already submitted.",
                record=failed,
                original=exc,
            ) from exc
        raise


def _run_estimator_job(
    estimator: Any,
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    *,
    repeat_index: int = 0,
    job_observer: Callable[[dict[str, Any]], None] | None = None,
    job_context: Mapping[str, Any] | None = None,
) -> EstimatorExecutionResult:
    errors: list[Exception] = []

    # V2-style tuple(pub) invocation
    for call_path, pub in (
        ("v2_single_observable", [(circuit, observable)]),
        ("v2_list_observable", [(circuit, [observable])]),
    ):
        job = None
        try:
            job = estimator.run(pub)
            submitted = _runtime_job_record_from_job(
                job,
                repeat_index=int(repeat_index),
                call_path=str(call_path),
            )
            _emit_runtime_job_event(
                job_observer,
                event="submitted",
                record=submitted,
                context=job_context,
            )
            result = job.result()
            value = float(np.real(_extract_expectation_value(result)))
            completed = _runtime_job_record_from_job(
                job,
                repeat_index=int(repeat_index),
                call_path=str(call_path),
                expectation_value=value,
            )
            _emit_runtime_job_event(
                job_observer,
                event="completed",
                record=completed,
                context=job_context,
            )
            return EstimatorExecutionResult(
                expectation_value=value,
                job_records=(completed,),
                used_call_path=str(call_path),
            )
        except Exception as exc:
            if job is not None:
                failed = _runtime_job_record_from_job(
                    job,
                    repeat_index=int(repeat_index),
                    call_path=str(call_path),
                    error=f"{type(exc).__name__}: {exc}",
                )
                _emit_runtime_job_event(
                    job_observer,
                    event="failed",
                    record=failed,
                    context=job_context,
                )
                raise SubmittedRuntimeJobError(
                    "Estimator execution failed after a Runtime job was already submitted. "
                    "Refusing alternate call-path retries to avoid duplicate hardware jobs.",
                    record=failed,
                    original=exc,
                ) from exc
            errors.append(exc)

    supports_v1_style = True
    try:
        run_sig = inspect.signature(estimator.run)
        positional_names = [
            p.name
            for p in run_sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        supports_v1_style = "observables" in positional_names or len(positional_names) >= 2
    except Exception:
        supports_v1_style = True

    # V1-style invocation
    if supports_v1_style:
        job = None
        try:
            job = estimator.run([circuit], [observable])
            submitted = _runtime_job_record_from_job(
                job,
                repeat_index=int(repeat_index),
                call_path="v1_lists",
            )
            _emit_runtime_job_event(
                job_observer,
                event="submitted",
                record=submitted,
                context=job_context,
            )
            result = job.result()
            value = float(np.real(_extract_expectation_value(result)))
            completed = _runtime_job_record_from_job(
                job,
                repeat_index=int(repeat_index),
                call_path="v1_lists",
                expectation_value=value,
            )
            _emit_runtime_job_event(
                job_observer,
                event="completed",
                record=completed,
                context=job_context,
            )
            return EstimatorExecutionResult(
                expectation_value=value,
                job_records=(completed,),
                used_call_path="v1_lists",
            )
        except Exception as exc:
            if job is not None:
                failed = _runtime_job_record_from_job(
                    job,
                    repeat_index=int(repeat_index),
                    call_path="v1_lists",
                    error=f"{type(exc).__name__}: {exc}",
                )
                _emit_runtime_job_event(
                    job_observer,
                    event="failed",
                    record=failed,
                    context=job_context,
                )
                raise SubmittedRuntimeJobError(
                    "Estimator execution failed after a Runtime job was already submitted. "
                    "Refusing alternate call-path retries to avoid duplicate hardware jobs.",
                    record=failed,
                    original=exc,
                ) from exc
            errors.append(exc)

    msg = "; ".join(f"{type(e).__name__}: {e}" for e in errors)
    raise RuntimeError(f"Estimator execution failed across known call paths. Details: {msg}")


def _run_sampler_job(
    sampler: Any,
    circuit: QuantumCircuit,
    *,
    shots: int | None,
    repeat_index: int = 0,
    job_observer: Callable[[dict[str, Any]], None] | None = None,
    job_context: Mapping[str, Any] | None = None,
) -> SamplerExecutionResult:
    job = None
    try:
        job = sampler.run([circuit], shots=(None if shots is None else int(shots)))
        submitted = _runtime_job_record_from_job(
            job,
            repeat_index=int(repeat_index),
            call_path="sampler_v2_single_pub",
        )
        _emit_runtime_job_event(
            job_observer,
            event="submitted",
            record=submitted,
            context=job_context,
        )
        result = job.result()
        counts = _extract_sampler_counts(result)
        completed = _runtime_job_record_from_job(
            job,
            repeat_index=int(repeat_index),
            call_path="sampler_v2_single_pub",
        )
        _emit_runtime_job_event(
            job_observer,
            event="completed",
            record=completed,
            context=job_context,
        )
        return SamplerExecutionResult(
            counts={str(bitstr): int(ct) for bitstr, ct in dict(counts).items()},
            job_records=(completed,),
            used_call_path="sampler_v2_single_pub",
        )
    except Exception as exc:
        if job is not None:
            failed = _runtime_job_record_from_job(
                job,
                repeat_index=int(repeat_index),
                call_path="sampler_v2_single_pub",
                error=f"{type(exc).__name__}: {exc}",
            )
            _emit_runtime_job_event(
                job_observer,
                event="failed",
                record=failed,
                context=job_context,
            )
            raise SubmittedRuntimeJobError(
                "Sampler execution failed after a Runtime job was already submitted.",
                record=failed,
                original=exc,
            ) from exc
        raise


def _fetch_runtime_job_record(
    job_id: str,
    *,
    require_result: bool,
) -> RuntimeJobRecord:
    from qiskit_ibm_runtime import QiskitRuntimeService

    job_id_s = str(job_id).strip()
    if not job_id_s:
        raise ValueError("job_id must be non-empty for runtime job refresh.")
    try:
        svc = QiskitRuntimeService()
        job = svc.job(job_id_s)
    except Exception as exc:  # pragma: no cover - network/runtime variability
        return RuntimeJobRecord(
            job_id=job_id_s,
            repeat_index=0,
            call_path="recovered_job_lookup",
            status="lookup_failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    value = None
    error = None
    status_str = "unknown"
    try:
        status_obj = job.status()
        status_str = str(getattr(status_obj, "name", status_obj))
    except Exception as exc:  # pragma: no cover - network/runtime variability
        error = f"{type(exc).__name__}: {exc}"

    terminal_success = {"DONE", "COMPLETED"}
    if bool(require_result) and str(status_str).upper() in terminal_success:
        try:
            result = job.result()
            value = float(np.real(_extract_expectation_value(result)))
        except Exception as exc:  # pragma: no cover - network/runtime variability
            error = f"{type(exc).__name__}: {exc}"
    elif bool(require_result) and str(status_str).upper() not in terminal_success and error is None:
        error = f"Skipped result lookup because job status is {status_str}."
    return _runtime_job_record_from_job(
        job,
        repeat_index=0,
        call_path="recovered_job_lookup",
        expectation_value=value,
        error=error,
    )


def _term_measurement_circuit(base: QuantumCircuit, pauli_label_ixyz: str) -> QuantumCircuit:
    """Rotate into Pauli measurement basis and measure all qubits."""
    label = str(pauli_label_ixyz).upper()
    n = int(base.num_qubits)
    if len(label) != n:
        raise ValueError(f"Pauli label length {len(label)} does not match circuit qubits {n}")

    qc = base.copy()
    for q in range(n):
        op = label[n - 1 - q]  # left-to-right is q_(n-1)..q_0; q0 rightmost
        if op == "X":
            qc.h(q)
        elif op == "Y":
            qc.sdg(q)
            qc.h(q)
        elif op in {"I", "Z"}:
            continue
        else:
            raise ValueError(f"Unsupported Pauli op '{op}' in '{label}'")
    qc.measure_all()
    return qc


def _sparse_term_measurement_circuit(
    base: QuantumCircuit,
    pauli_label_ixyz: str,
) -> tuple[QuantumCircuit, tuple[int, ...]]:
    """Rotate into Pauli measurement basis and measure only active qubits."""
    label = str(pauli_label_ixyz).upper()
    n = int(base.num_qubits)
    if len(label) != n:
        raise ValueError(f"Pauli label length {len(label)} does not match circuit qubits {n}")

    active_logical = _active_logical_qubits_for_label(label)
    qc = base.copy()
    if active_logical:
        creg = ClassicalRegister(len(active_logical), "m")
        qc.add_register(creg)
        for q in active_logical:
            op = label[n - 1 - q]
            if op == "X":
                qc.h(q)
            elif op == "Y":
                qc.sdg(q)
                qc.h(q)
            elif op in {"I", "Z"}:
                continue
            else:
                raise ValueError(f"Unsupported Pauli op '{op}' in '{label}'")
        for idx, q in enumerate(active_logical):
            qc.measure(int(q), creg[int(idx)])
    return qc, tuple(int(q) for q in active_logical)


def _pauli_parity_from_bitstring(bitstr_raw: str, pauli_label_ixyz: str, n_qubits: int) -> float:
    label = str(pauli_label_ixyz).upper()
    if len(label) != int(n_qubits):
        raise ValueError(f"Pauli label length {len(label)} does not match n_qubits={n_qubits}")
    bitstr = str(bitstr_raw).replace(" ", "")
    if len(bitstr) < int(n_qubits):
        bitstr = bitstr.zfill(int(n_qubits))
    parity = 1.0
    for q in range(int(n_qubits)):
        if label[int(n_qubits) - 1 - q] == "I":
            continue
        bit = bitstr[-1 - int(q)]
        parity *= (-1.0 if bit == "1" else 1.0)
    return float(parity)


def _pauli_expectation_from_counts(counts: dict[str, int], pauli_label_ixyz: str, n_qubits: int) -> float:
    label = str(pauli_label_ixyz).upper()
    if len(label) != int(n_qubits):
        raise ValueError(f"Pauli label length {len(label)} does not match n_qubits={n_qubits}")
    active_q = [q for q in range(int(n_qubits)) if label[int(n_qubits) - 1 - q] != "I"]
    if not active_q:
        return 1.0
    shots = int(sum(int(v) for v in counts.values()))
    if shots <= 0:
        raise RuntimeError("Sampler returned zero total shots.")

    acc = 0.0
    for bitstr_raw, ct in counts.items():
        acc += _pauli_parity_from_bitstring(str(bitstr_raw), label, int(n_qubits)) * float(ct)
    return float(acc / float(shots))


def _observable_is_diagonal(observable: SparsePauliOp) -> bool:
    for label, _coeff in observable.to_list():
        if any(ch not in {"I", "Z"} for ch in str(label).upper()):
            return False
    return True


def _spin_orbital_index_sets(num_sites: int, ordering: str) -> tuple[list[int], list[int]]:
    if str(ordering).strip().lower() == "interleaved":
        return [2 * i for i in range(int(num_sites))], [2 * i + 1 for i in range(int(num_sites))]
    return list(range(int(num_sites))), list(range(int(num_sites), 2 * int(num_sites)))


def _hh_fermion_sector_weights(
    psi: np.ndarray,
    *,
    num_sites: int,
    ordering: str,
) -> dict[tuple[int, int], float]:
    alpha_indices, beta_indices = _spin_orbital_index_sets(int(num_sites), str(ordering))
    out: dict[tuple[int, int], float] = {}
    for idx, amp in enumerate(np.asarray(psi, dtype=complex).reshape(-1)):
        prob = float(abs(complex(amp)) ** 2)
        if prob <= 1e-14:
            continue
        n_up = int(sum((int(idx) >> int(q)) & 1 for q in alpha_indices))
        n_dn = int(sum((int(idx) >> int(q)) & 1 for q in beta_indices))
        key = (int(n_up), int(n_dn))
        out[key] = float(out.get(key, 0.0) + prob)
    return out


def _summarize_hh_exact_diagonal_reference(
    circuit: QuantumCircuit,
    *,
    num_sites: int,
    ordering: str,
    sector_n_up: int,
    sector_n_dn: int,
) -> dict[str, Any]:
    ordering_resolved = str(ordering).strip().lower() or "blocked"
    num_sites_resolved = int(num_sites)
    n_qubits = int(circuit.num_qubits)
    if num_sites_resolved <= 0:
        raise ValueError("HH exact diagonal reference requires num_sites > 0.")
    if n_qubits < 2 * int(num_sites_resolved):
        raise ValueError(
            "HH exact diagonal reference requires num_qubits to cover the full fermion register."
        )
    statevector = Statevector.from_instruction(circuit)
    psi = np.asarray(statevector.data, dtype=complex).reshape(-1)
    sector_weights = _hh_fermion_sector_weights(
        psi,
        num_sites=int(num_sites_resolved),
        ordering=str(ordering_resolved),
    )
    doublon_qop = _hh_total_doublon_qop(
        num_qubits=int(n_qubits),
        num_sites=int(num_sites_resolved),
        ordering=str(ordering_resolved),
    )
    doublon_total = float(np.real(statevector.expectation_value(doublon_qop)))
    sector_key = (int(sector_n_up), int(sector_n_dn))
    return {
        "source": "ideal_diagonal_v1",
        "available": True,
        "num_qubits": int(n_qubits),
        "num_sites": int(num_sites_resolved),
        "ordering": str(ordering_resolved),
        "target_sector": {
            "n_up": int(sector_key[0]),
            "n_dn": int(sector_key[1]),
        },
        "sector_weight": float(sector_weights.get(sector_key, 0.0)),
        "doublon_total": float(doublon_total),
    }


def _bitstring_passes_sector(
    bitstr_raw: str,
    *,
    n_qubits: int,
    num_sites: int,
    ordering: str,
    sector_n_up: int,
    sector_n_dn: int,
) -> bool:
    bitstr = str(bitstr_raw).replace(" ", "")
    if len(bitstr) < int(n_qubits):
        bitstr = bitstr.zfill(int(n_qubits))
    alpha_indices, beta_indices = _spin_orbital_index_sets(int(num_sites), str(ordering))
    n_up = sum(1 for idx in alpha_indices if bitstr[-1 - int(idx)] == "1")
    n_dn = sum(1 for idx in beta_indices if bitstr[-1 - int(idx)] == "1")
    return int(n_up) == int(sector_n_up) and int(n_dn) == int(sector_n_dn)


def _diagonal_expectation_from_counts(counts: dict[str, int], observable: SparsePauliOp) -> float:
    total = 0.0 + 0.0j
    n = int(observable.num_qubits)
    for label, coeff in observable.to_list():
        total += complex(coeff) * complex(_pauli_expectation_from_counts(counts, str(label), n), 0.0)
    return float(np.real(total))


def _exact_postselected_diagonal_expectation(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    symmetry_cfg: Mapping[str, Any],
) -> tuple[float, float]:
    psi = np.asarray(Statevector.from_instruction(circuit).data, dtype=complex).reshape(-1)
    n_qubits = int(circuit.num_qubits)
    kept_prob = 0.0
    total = 0.0 + 0.0j
    for idx, amp in enumerate(psi):
        prob = float(abs(complex(amp)) ** 2)
        if prob <= 1e-18:
            continue
        bitstr = format(int(idx), f"0{n_qubits}b")
        if not _bitstring_passes_sector(
            bitstr,
            n_qubits=int(n_qubits),
            num_sites=int(symmetry_cfg.get("num_sites", 0)),
            ordering=str(symmetry_cfg.get("ordering", "blocked")),
            sector_n_up=int(symmetry_cfg.get("sector_n_up", 0)),
            sector_n_dn=int(symmetry_cfg.get("sector_n_dn", 0)),
        ):
            continue
        kept_prob += prob
        for label, coeff in observable.to_list():
            total += complex(coeff) * complex(
                _pauli_parity_from_bitstring(bitstr, str(label), int(n_qubits)) * prob,
                0.0,
            )
    if kept_prob <= 0.0:
        raise RuntimeError("Symmetry postselection retained zero probability mass.")
    return float(np.real(total) / kept_prob), float(kept_prob)


def _exact_projector_renorm_diagonal_expectation(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    symmetry_cfg: Mapping[str, Any],
) -> tuple[float, float]:
    psi = np.asarray(Statevector.from_instruction(circuit).data, dtype=complex).reshape(-1)
    n_qubits = int(circuit.num_qubits)
    sector_prob = 0.0
    numerator = 0.0 + 0.0j
    for idx, amp in enumerate(psi):
        prob = float(abs(complex(amp)) ** 2)
        if prob <= 1e-18:
            continue
        bitstr = format(int(idx), f"0{n_qubits}b")
        in_sector = _bitstring_passes_sector(
            bitstr,
            n_qubits=int(n_qubits),
            num_sites=int(symmetry_cfg.get("num_sites", 0)),
            ordering=str(symmetry_cfg.get("ordering", "blocked")),
            sector_n_up=int(symmetry_cfg.get("sector_n_up", 0)),
            sector_n_dn=int(symmetry_cfg.get("sector_n_dn", 0)),
        )
        if not in_sector:
            continue
        sector_prob += prob
        for label, coeff in observable.to_list():
            numerator += complex(coeff) * complex(
                _pauli_parity_from_bitstring(bitstr, str(label), int(n_qubits)) * prob,
                0.0,
            )
    if sector_prob <= 0.0:
        raise RuntimeError("Projector renormalization retained zero probability mass.")
    return float(np.real(numerator) / sector_prob), float(sector_prob)


def _sample_measurement_counts(
    circuit: QuantumCircuit,
    cfg: OracleConfig,
    *,
    repeat_idx: int,
) -> dict[str, int]:
    measured = circuit.copy()
    measured.measure_all()
    mode = str(cfg.noise_mode).strip().lower()
    if mode == "shots" or mode == "ideal":
        from qiskit.primitives import StatevectorSampler

        sampler = StatevectorSampler(
            default_shots=int(cfg.shots),
            seed=int(cfg.seed) + int(repeat_idx),
        )
        job = sampler.run([measured])
        result = job.result()
        return dict(result[0].join_data().get_counts())

    if mode == "aer_noise":
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel

        backend_obj, _backend_name, _using_fake = _resolve_noise_backend(cfg)
        noise_model = NoiseModel.from_backend(backend_obj)
        sim = AerSimulator(noise_model=noise_model, seed_simulator=int(cfg.seed) + int(repeat_idx))
        compiled = transpile(measured, sim, optimization_level=0)
        result = sim.run(compiled, shots=int(cfg.shots)).result()
        counts = result.get_counts()
        if isinstance(counts, list):
            counts = counts[0]
        return dict(counts)

    raise RuntimeError(f"Counts-based symmetry mitigation is unavailable for noise_mode={mode!r}.")


def _apply_observable_layout_for_compiled_circuit(
    observable: SparsePauliOp,
    compiled_circuit: QuantumCircuit,
) -> SparsePauliOp:
    layout = getattr(compiled_circuit, "layout", None)
    if layout is None:
        return observable
    try:
        return observable.apply_layout(layout).simplify(atol=1e-12)
    except Exception as exc:
        raise RuntimeError(
            "Failed to apply transpiled circuit layout to observable for Runtime execution."
        ) from exc


def _logical_to_physical_qubits(compiled: QuantumCircuit, logical_qubits: int) -> tuple[int, ...]:
    layout = getattr(compiled, "layout", None)
    if layout is None or not hasattr(layout, "final_index_layout"):
        return tuple(range(int(logical_qubits)))
    try:
        mapped = list(layout.final_index_layout())
    except Exception:
        mapped = []
    if len(mapped) < int(logical_qubits):
        return tuple(range(int(logical_qubits)))
    return tuple(int(mapped[idx]) for idx in range(int(logical_qubits)))


def _active_logical_qubits_for_label(pauli_label_ixyz: str) -> tuple[int, ...]:
    label = str(pauli_label_ixyz).upper()
    n = int(len(label))
    return tuple(q for q in range(n) if label[n - 1 - q] != "I")


def _active_pauli_weight(pauli_label_ixyz: str) -> int:
    return int(sum(1 for ch in str(pauli_label_ixyz).upper() if ch != "I"))


def _parity_expectation_from_active_counts(counts: Mapping[str, int], num_bits: int) -> float:
    k = int(num_bits)
    if k <= 0:
        return 1.0
    shots = int(sum(int(v) for v in counts.values()))
    if shots <= 0:
        raise RuntimeError("Sampler returned zero total shots.")
    acc = 0.0
    for bitstr_raw, ct in counts.items():
        bitstr = str(bitstr_raw).replace(" ", "")
        if len(bitstr) < k:
            bitstr = bitstr.zfill(k)
        ones = sum(1 for ch in bitstr[-k:] if ch == "1")
        parity = -1.0 if (ones % 2) else 1.0
        acc += float(parity) * float(ct)
    return float(acc / float(shots))


def _parity_expectation_from_quasi(quasi: Mapping[str, float], num_bits: int) -> float:
    k = int(num_bits)
    if k <= 0:
        return 1.0
    total = 0.0
    for bitstr_raw, prob in quasi.items():
        bitstr = str(bitstr_raw).replace(" ", "")
        if len(bitstr) < k:
            bitstr = bitstr.zfill(k)
        ones = sum(1 for ch in bitstr[-k:] if ch == "1")
        parity = -1.0 if (ones % 2) else 1.0
        total += float(parity) * float(prob)
    return float(total)


def _negative_quasi_mass(quasi: Mapping[str, Any]) -> float:
    total = 0.0
    for value in quasi.values():
        val = float(np.real(value))
        if val < 0.0:
            total += float(-val)
    return float(total)


def _resolve_mthree() -> Any:
    try:
        import mthree  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "readout mitigation strategy 'mthree' requires the optional dependency `mthree`."
        ) from exc
    return mthree


def _ensure_mthree_calibration_cached(
    *,
    backend_target: Any | None,
    mitigator: Any | None,
    calibrated_qubits: set[tuple[int, ...]],
    active_physical_qubits: Sequence[int],
    shots: int,
    mthree_module: Any | None = None,
) -> tuple[Any, bool]:
    qubits = tuple(int(q) for q in active_physical_qubits)
    if qubits in calibrated_qubits:
        return mitigator, False
    if backend_target is None:
        raise RuntimeError("backend target is unavailable for mthree calibration.")
    resolved_mitigator = mitigator
    if resolved_mitigator is None:
        module = _resolve_mthree() if mthree_module is None else mthree_module
        resolved_mitigator = module.M3Mitigation(backend_target)
    resolved_mitigator.cals_from_system(
        qubits=list(qubits),
        shots=int(shots),
        async_cal=False,
    )
    calibrated_qubits.add(qubits)
    return resolved_mitigator, True


def _copy_noise_model_components(
    source_model: Any,
    *,
    include_quantum: bool,
    include_readout: bool,
) -> Any:
    from qiskit_aer.noise import NoiseModel

    basis_gates = list(getattr(source_model, "basis_gates", []) or [])
    model = NoiseModel(basis_gates=(basis_gates or None))

    if include_quantum:
        for inst_name, error in getattr(source_model, "_default_quantum_errors", {}).items():
            model.add_all_qubit_quantum_error(error, str(inst_name))
        for inst_name, qubit_map in getattr(source_model, "_local_quantum_errors", {}).items():
            if not isinstance(qubit_map, Mapping):
                continue
            for qubits, error in qubit_map.items():
                model.add_quantum_error(error, str(inst_name), [int(q) for q in tuple(qubits)])

    if include_readout:
        default_readout = getattr(source_model, "_default_readout_error", None)
        if default_readout is not None:
            model.add_all_qubit_readout_error(default_readout)
        for qubits, error in getattr(source_model, "_local_readout_errors", {}).items():
            model.add_readout_error(error, [int(q) for q in tuple(qubits)])

    return model


def _apply_mthree_readout_correction(
    *,
    mitigator: Any,
    counts: Mapping[str, int],
    active_physical_qubits: Sequence[int],
) -> tuple[dict[str, float], dict[str, Any]]:
    raw = mitigator.apply_correction(
        dict(counts),
        qubits=[int(q) for q in active_physical_qubits],
        details=True,
        return_mitigation_overhead=True,
    )
    if isinstance(raw, tuple):
        quasi, details = raw
    else:
        quasi, details = raw, {}
    quasi_map = {str(k): float(np.real(v)) for k, v in dict(quasi).items()}
    out_details = dict(details) if isinstance(details, Mapping) else {}
    out_details["mitigation_overhead"] = float(getattr(quasi, "mitigation_overhead", float("nan")))
    out_details["shots"] = int(getattr(quasi, "shots", int(sum(int(v) for v in counts.values()))))
    out_details["negative_mass"] = _negative_quasi_mass(quasi_map)
    return quasi_map, out_details


def _local_gate_twirling_enabled(mitigation_cfg: Mapping[str, Any]) -> bool:
    return bool(mitigation_cfg.get("local_gate_twirling", False))


def _twirl_compiled_two_qubit_base(
    compiled_base: QuantumCircuit,
    *,
    backend_target: Any | None,
    seed: int,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    base_metrics = _compiled_metrics_payload(compiled_base)
    base_two_qubit = int(base_metrics.get("compiled_two_qubit_count", 0))
    if base_two_qubit <= 0:
        return compiled_base, {
            "requested": True,
            "applied": False,
            "reason": "no_two_qubit_gates",
            "engine": "qiskit.circuit.pauli_twirl_2q_gates",
            "seed": int(seed),
            "gate_set": ["cx", "cz", "ecr", "iswap"],
            "base_metrics": dict(base_metrics),
            "twirled_metrics": dict(base_metrics),
        }

    twirled = pauli_twirl_2q_gates(
        compiled_base,
        seed=int(seed),
        target=(None if backend_target is None else getattr(backend_target, "target", None)),
    )
    if isinstance(twirled, list):
        if not twirled:
            raise RuntimeError("Local gate twirling returned an empty circuit list.")
        twirled_circuit = twirled[0]
    else:
        twirled_circuit = twirled
    twirled_metrics = _compiled_metrics_payload(twirled_circuit)
    return twirled_circuit, {
        "requested": True,
        "applied": True,
        "reason": "",
        "engine": "qiskit.circuit.pauli_twirl_2q_gates",
        "seed": int(seed),
        "gate_set": ["cx", "cz", "ecr", "iswap"],
        "base_metrics": dict(base_metrics),
        "twirled_metrics": dict(twirled_metrics),
    }


def _postselected_counts_and_fraction(
    counts: Mapping[str, int],
    *,
    n_qubits: int,
    symmetry_cfg: Mapping[str, Any],
) -> tuple[dict[str, int], float]:
    kept: dict[str, int] = {}
    total = int(sum(int(v) for v in counts.values()))
    if total <= 0:
        raise RuntimeError("Counts-based symmetry mitigation received zero total shots.")
    kept_total = 0
    for bitstr_raw, ct_raw in counts.items():
        ct = int(ct_raw)
        if ct <= 0:
            continue
        if _bitstring_passes_sector(
            str(bitstr_raw),
            n_qubits=int(n_qubits),
            num_sites=int(symmetry_cfg.get("num_sites", 0)),
            ordering=str(symmetry_cfg.get("ordering", "blocked")),
            sector_n_up=int(symmetry_cfg.get("sector_n_up", 0)),
            sector_n_dn=int(symmetry_cfg.get("sector_n_dn", 0)),
        ):
            kept[str(bitstr_raw)] = kept.get(str(bitstr_raw), 0) + int(ct)
            kept_total += int(ct)
    if kept_total <= 0:
        raise RuntimeError("Symmetry postselection retained zero shots.")
    return kept, float(kept_total) / float(total)


def _projector_renorm_diagonal_expectation_from_counts(
    counts: Mapping[str, int],
    observable: SparsePauliOp,
    *,
    n_qubits: int,
    symmetry_cfg: Mapping[str, Any],
) -> tuple[float, float]:
    total_shots = int(sum(int(v) for v in counts.values()))
    if total_shots <= 0:
        raise RuntimeError("Counts-based projector renormalization received zero total shots.")
    sector_shots = 0
    numerator = 0.0 + 0.0j
    for bitstr_raw, ct_raw in counts.items():
        ct = int(ct_raw)
        if ct <= 0:
            continue
        if not _bitstring_passes_sector(
            str(bitstr_raw),
            n_qubits=int(n_qubits),
            num_sites=int(symmetry_cfg.get("num_sites", 0)),
            ordering=str(symmetry_cfg.get("ordering", "blocked")),
            sector_n_up=int(symmetry_cfg.get("sector_n_up", 0)),
            sector_n_dn=int(symmetry_cfg.get("sector_n_dn", 0)),
        ):
            continue
        sector_shots += int(ct)
        for label, coeff in observable.to_list():
            numerator += complex(coeff) * complex(
                _pauli_parity_from_bitstring(str(bitstr_raw), str(label), int(n_qubits)) * float(ct),
                0.0,
            )
    if sector_shots <= 0:
        raise RuntimeError("Projector renormalization retained zero shots.")
    sector_prob = float(sector_shots) / float(total_shots)
    numerator_expectation = numerator / float(total_shots)
    return float(np.real(numerator_expectation) / sector_prob), float(sector_prob)


def _raw_record_view(record: RawMeasurementRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, RawMeasurementRecord):
        raw: dict[str, Any] = {
            "evaluation_id": str(record.evaluation_id),
            "observable_family": str(record.observable_family),
            "basis_label": str(record.basis_label).upper(),
            "num_qubits": int(record.num_qubits),
            "measured_logical_qubits": tuple(int(x) for x in record.measured_logical_qubits),
            "measured_physical_qubits": (
                None
                if record.measured_physical_qubits is None
                else tuple(int(x) for x in record.measured_physical_qubits)
            ),
            "repeat_index": int(record.repeat_index),
            "counts": {str(k): int(v) for k, v in dict(record.counts).items()},
            "shots_completed": int(record.shots_completed),
            "semantic_tags": dict(record.semantic_tags),
            "transport": str(record.transport),
            "compile_signature": dict(record.compile_signature),
        }
    elif isinstance(record, Mapping):
        counts = {str(k): int(v) for k, v in dict(record.get("counts", {})).items()}
        shots_completed = int(record.get("shots_completed", sum(int(v) for v in counts.values())))
        measured_logical = tuple(int(x) for x in tuple(record.get("measured_logical_qubits", ())))
        basis_label = str(record.get("basis_label", "")).replace(" ", "").upper()
        num_qubits_raw = record.get("num_qubits", None)
        num_qubits = (
            int(num_qubits_raw)
            if num_qubits_raw is not None
            else (len(basis_label) if basis_label else len(measured_logical))
        )
        semantic_tags_raw = record.get("semantic_tags", {})
        compile_signature_raw = record.get("compile_signature", {})
        measured_physical_raw = record.get("measured_physical_qubits", None)
        raw = {
            "evaluation_id": str(record.get("evaluation_id", "")),
            "observable_family": (
                None
                if record.get("observable_family", None) in {None, ""}
                else str(record.get("observable_family"))
            ),
            "basis_label": str(basis_label),
            "num_qubits": int(num_qubits),
            "measured_logical_qubits": measured_logical,
            "measured_physical_qubits": (
                None
                if measured_physical_raw is None or tuple(measured_physical_raw) == ()
                else tuple(int(x) for x in tuple(measured_physical_raw))
            ),
            "repeat_index": int(record.get("repeat_index", 0)),
            "counts": dict(counts),
            "shots_completed": int(shots_completed),
            "semantic_tags": (
                dict(semantic_tags_raw) if isinstance(semantic_tags_raw, Mapping) else {}
            ),
            "transport": str(record.get("transport", "")),
            "compile_signature": (
                dict(compile_signature_raw) if isinstance(compile_signature_raw, Mapping) else {}
            ),
        }
    else:
        raise TypeError("raw record summary expects RawMeasurementRecord or mapping rows.")
    return raw


def _resolve_measured_physical_qubits_from_raw_view(row: Mapping[str, Any]) -> tuple[int, ...] | None:
    explicit = row.get("measured_physical_qubits", None)
    if explicit is not None and tuple(explicit) != ():
        return tuple(int(x) for x in tuple(explicit))
    compile_signature = row.get("compile_signature", {})
    if isinstance(compile_signature, Mapping):
        layout = compile_signature.get("layout_physical_qubits", None)
        if layout is not None and tuple(layout) != ():
            return tuple(int(x) for x in tuple(layout))
    return None


def _normalize_bitstring_weight_map(
    weight_map: Mapping[str, float | int],
    *,
    n_qubits: int,
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    total = 0.0
    for bitstr_raw, weight_raw in weight_map.items():
        bitstr = str(bitstr_raw).replace(" ", "")
        if len(bitstr) < int(n_qubits):
            bitstr = bitstr.zfill(int(n_qubits))
        if len(bitstr) != int(n_qubits):
            raise ValueError(
                f"Bitstring length mismatch in weighted map: got {len(bitstr)}, expected {int(n_qubits)}."
            )
        weight = float(np.real(weight_raw))
        normalized[bitstr] = float(normalized.get(bitstr, 0.0) + weight)
        total += float(weight)
    if (not np.isfinite(total)) or total <= 0.0:
        raise RuntimeError("Weighted bitstring map has non-positive total mass.")
    inv_total = 1.0 / float(total)
    return {bitstr: float(weight * inv_total) for bitstr, weight in normalized.items()}


def _diagonal_expectation_from_weight_map(
    weight_map: Mapping[str, float],
    observable: SparsePauliOp,
    *,
    n_qubits: int,
) -> float:
    total = 0.0 + 0.0j
    for bitstr_raw, weight_raw in weight_map.items():
        bitstr = str(bitstr_raw).replace(" ", "")
        if len(bitstr) < int(n_qubits):
            bitstr = bitstr.zfill(int(n_qubits))
        if len(bitstr) != int(n_qubits):
            raise ValueError(
                f"Bitstring length mismatch in diagonal weighted expectation: got {len(bitstr)}, expected {int(n_qubits)}."
            )
        weight = float(weight_raw)
        for label, coeff in observable.to_list():
            total += complex(coeff) * complex(
                _pauli_parity_from_bitstring(bitstr, str(label), int(n_qubits)) * weight,
                0.0,
            )
    return float(np.real(total))


def _postselected_weight_map_and_fraction(
    weight_map: Mapping[str, float],
    *,
    n_qubits: int,
    symmetry_cfg: Mapping[str, Any],
) -> tuple[dict[str, float], float]:
    kept: dict[str, float] = {}
    kept_mass = 0.0
    for bitstr_raw, weight_raw in weight_map.items():
        bitstr = str(bitstr_raw).replace(" ", "")
        if len(bitstr) < int(n_qubits):
            bitstr = bitstr.zfill(int(n_qubits))
        if len(bitstr) != int(n_qubits):
            raise ValueError(
                f"Bitstring length mismatch in weighted symmetry mitigation: got {len(bitstr)}, expected {int(n_qubits)}."
            )
        weight = float(weight_raw)
        if _bitstring_passes_sector(
            bitstr,
            n_qubits=int(n_qubits),
            num_sites=int(symmetry_cfg.get("num_sites", 0)),
            ordering=str(symmetry_cfg.get("ordering", "blocked")),
            sector_n_up=int(symmetry_cfg.get("sector_n_up", 0)),
            sector_n_dn=int(symmetry_cfg.get("sector_n_dn", 0)),
        ):
            kept[bitstr] = float(kept.get(bitstr, 0.0) + weight)
            kept_mass += float(weight)
    if kept_mass <= 0.0:
        raise RuntimeError("Symmetry postselection retained zero probability mass.")
    inv_kept = 1.0 / float(kept_mass)
    return (
        {bitstr: float(weight * inv_kept) for bitstr, weight in kept.items()},
        float(kept_mass),
    )


def _summarize_hh_full_register_z_records_postprocessed(
    records: Sequence[RawMeasurementRecord | Mapping[str, Any]],
    *,
    backend_target: Any | None,
    mitigation_config: Mapping[str, Any],
    symmetry_mitigation_config: Mapping[str, Any],
    num_sites: int | None = None,
    ordering: str | None = None,
    sector_n_up: int | None = None,
    sector_n_dn: int | None = None,
    expected_repeat_count: int | None = None,
    shots: int | None = None,
) -> dict[str, Any]:
    try:
        if len(records) <= 0:
            raise ValueError("HH postprocessed symmetry summary requires at least one raw record.")
        views = [_raw_record_view(record) for record in records]
        first = dict(views[0])
        semantic_tags = dict(first.get("semantic_tags", {}))
        eval_id = str(first.get("evaluation_id", ""))
        if eval_id == "":
            raise ValueError("HH postprocessed symmetry summary requires a non-empty evaluation_id.")
        observable_family = first.get("observable_family", None)
        num_qubits = int(first.get("num_qubits", 0))
        if num_qubits <= 0:
            raise ValueError("HH postprocessed symmetry summary requires num_qubits > 0.")

        def _resolve_int(explicit: int | None, tag_key: str) -> int | None:
            if explicit is not None:
                return int(explicit)
            raw = semantic_tags.get(tag_key, None)
            if raw in {None, ""}:
                return None
            return int(raw)

        num_sites_resolved = _resolve_int(num_sites, "symmetry_num_sites")
        sector_n_up_resolved = _resolve_int(sector_n_up, "symmetry_sector_n_up")
        sector_n_dn_resolved = _resolve_int(sector_n_dn, "symmetry_sector_n_dn")
        ordering_resolved = (
            str(ordering)
            if ordering not in {None, ""}
            else str(semantic_tags.get("symmetry_ordering", "blocked"))
        ).strip().lower() or "blocked"

        if num_sites_resolved is None:
            raise ValueError("HH postprocessed symmetry summary requires num_sites metadata.")
        if sector_n_up_resolved is None or sector_n_dn_resolved is None:
            raise ValueError("HH postprocessed symmetry summary requires target-sector metadata.")
        if int(num_qubits) < 2 * int(num_sites_resolved):
            raise ValueError(
                "HH postprocessed symmetry summary requires num_qubits to cover the full fermion register."
            )

        expected_measured = tuple(range(int(num_qubits)))
        seen_repeat_indices: set[int] = set()
        for row in views:
            if str(row.get("evaluation_id", "")) != str(eval_id):
                raise ValueError(
                    "HH postprocessed symmetry summary requires exactly one evaluation_id."
                )
            if int(row.get("num_qubits", 0)) != int(num_qubits):
                raise ValueError(
                    "HH postprocessed symmetry summary requires a consistent num_qubits across records."
                )
            if row.get("observable_family", None) != observable_family:
                raise ValueError(
                    "HH postprocessed symmetry summary requires a consistent observable_family."
                )
            if str(row.get("basis_label", "")).upper() != ("Z" * int(num_qubits)):
                raise ValueError(
                    "HH postprocessed symmetry summary requires full-register all-Z basis records only."
                )
            measured_logical = tuple(int(x) for x in tuple(row.get("measured_logical_qubits", ())))
            if measured_logical != expected_measured:
                raise ValueError(
                    "HH postprocessed symmetry summary requires full-register measured_logical_qubits coverage."
                )
            repeat_index = int(row.get("repeat_index", 0))
            if repeat_index in seen_repeat_indices:
                raise ValueError(
                    "HH postprocessed symmetry summary requires unique repeat_index values."
                )
            seen_repeat_indices.add(repeat_index)

        mitigation_cfg = dict(normalize_mitigation_config(mitigation_config))
        symmetry_cfg = dict(
            normalize_symmetry_mitigation_config(symmetry_mitigation_config)
        )
        symmetry_cfg.update(
            {
                "num_sites": int(num_sites_resolved),
                "ordering": str(ordering_resolved),
                "sector_n_up": int(sector_n_up_resolved),
                "sector_n_dn": int(sector_n_dn_resolved),
            }
        )
        mitigation_mode = str(mitigation_cfg.get("mode", "none"))
        requested_symmetry_mode = str(symmetry_cfg.get("mode", "off"))
        if mitigation_mode not in {"none", "readout"}:
            return {
                "success": False,
                "available": False,
                "reason": "unsupported_mitigation_mode",
                "summary": None,
                "readout_details": None,
                "error_type": None,
                "error_message": None,
            }
        if requested_symmetry_mode not in {
            "off",
            "verify_only",
            "postselect_diag_v1",
            "projector_renorm_v1",
        }:
            return {
                "success": False,
                "available": False,
                "reason": "unsupported_symmetry_mode",
                "summary": None,
                "readout_details": None,
                "error_type": None,
                "error_message": None,
            }
        if mitigation_mode == "readout" and str(
            mitigation_cfg.get("local_readout_strategy") or "mthree"
        ) != "mthree":
            return {
                "success": False,
                "available": False,
                "reason": "unsupported_readout_strategy",
                "summary": None,
                "readout_details": None,
                "error_type": None,
                "error_message": None,
            }
        if mitigation_mode == "readout" and backend_target is None:
            return {
                "success": False,
                "available": False,
                "reason": "backend_target_unavailable",
                "summary": None,
                "readout_details": None,
                "error_type": None,
                "error_message": None,
            }

        doublon_qop = _hh_total_doublon_qop(
            num_qubits=int(num_qubits),
            num_sites=int(num_sites_resolved),
            ordering=str(ordering_resolved),
        )
        alpha_indices, beta_indices = _spin_orbital_index_sets(
            int(num_sites_resolved),
            str(ordering_resolved),
        )
        site_index_pairs = [
            (int(alpha_indices[site]), int(beta_indices[site]))
            for site in range(int(num_sites_resolved))
        ]
        site_doublon_qops = [
            _doublon_site_qop(int(num_qubits), int(up_idx), int(dn_idx))
            for up_idx, dn_idx in site_index_pairs
        ]
        site_charge_qops = [
            (
                _number_operator_qop(int(num_qubits), int(up_idx))
                + _number_operator_qop(int(num_qubits), int(dn_idx))
            ).simplify(atol=1e-12)
            for up_idx, dn_idx in site_index_pairs
        ]

        def _sector_distribution_from_weight_map(
            weight_map: Mapping[str, float],
        ) -> dict[tuple[int, int], float]:
            weights: dict[tuple[int, int], float] = {}
            for bitstr_raw, prob_raw in weight_map.items():
                bitstr = str(bitstr_raw).replace(" ", "")
                if len(bitstr) < int(num_qubits):
                    bitstr = bitstr.zfill(int(num_qubits))
                n_up = int(sum(1 for idx in alpha_indices if bitstr[-1 - int(idx)] == "1"))
                n_dn = int(sum(1 for idx in beta_indices if bitstr[-1 - int(idx)] == "1"))
                key = (int(n_up), int(n_dn))
                weights[key] = float(weights.get(key, 0.0) + float(prob_raw))
            return weights

        mthree_module = None
        mitigator = None
        calibrated_qubits: set[tuple[int, ...]] = set()
        sector_weight_samples: list[float] = []
        retained_fraction_samples: list[float] = []
        doublon_total_samples: list[float] = []
        sector_distributions_by_repeat: list[dict[tuple[int, int], float]] = []
        site_doublon_samples: list[list[float]] = [[] for _ in range(int(num_sites_resolved))]
        site_charge_samples: list[list[float]] = [[] for _ in range(int(num_sites_resolved))]
        negative_mass_samples: list[float] = []
        mitigation_overhead_samples: list[float] = []
        active_qubits_seen: list[tuple[int, ...]] = []

        for row in sorted(views, key=lambda item: int(item["repeat_index"])):
            counts = {str(k): int(v) for k, v in dict(row["counts"]).items()}
            if mitigation_mode == "readout":
                measured_physical = _resolve_measured_physical_qubits_from_raw_view(row)
                if measured_physical in {None, ()}:
                    return {
                        "success": False,
                        "available": False,
                        "reason": "missing_measured_physical_qubits",
                        "summary": None,
                        "readout_details": None,
                        "error_type": None,
                        "error_message": None,
                    }
                if mthree_module is None:
                    mthree_module = _resolve_mthree()
                mitigator, _cache_miss = _ensure_mthree_calibration_cached(
                    backend_target=backend_target,
                    mitigator=mitigator,
                    calibrated_qubits=calibrated_qubits,
                    active_physical_qubits=measured_physical,
                    shots=(
                        int(shots)
                        if shots not in {None, 0}
                        else int(row.get("shots_completed", sum(int(v) for v in counts.values())))
                    ),
                    mthree_module=mthree_module,
                )
                quasi_map, mitigation_details = _apply_mthree_readout_correction(
                    mitigator=mitigator,
                    counts=counts,
                    active_physical_qubits=measured_physical,
                )
                base_weight_map = _normalize_bitstring_weight_map(
                    quasi_map,
                    n_qubits=int(num_qubits),
                )
                negative_mass_samples.append(
                    float(mitigation_details.get("negative_mass", 0.0))
                )
                mitigation_overhead = mitigation_details.get(
                    "mitigation_overhead", float("nan")
                )
                if np.isfinite(float(mitigation_overhead)):
                    mitigation_overhead_samples.append(float(mitigation_overhead))
                active_qubits_seen.append(tuple(int(q) for q in measured_physical))
            else:
                shots_completed = int(
                    row.get("shots_completed", sum(int(v) for v in counts.values()))
                )
                if shots_completed <= 0:
                    raise RuntimeError(
                        "HH postprocessed symmetry summary requires strictly positive shots per repeat."
                    )
                base_weight_map = _normalize_bitstring_weight_map(
                    {
                        str(bitstr): float(int(ct)) / float(shots_completed)
                        for bitstr, ct in counts.items()
                    },
                    n_qubits=int(num_qubits),
                )
            sector_dist = _sector_distribution_from_weight_map(base_weight_map)
            sector_weight = float(
                sector_dist.get((int(sector_n_up_resolved), int(sector_n_dn_resolved)), 0.0)
            )
            sector_weight_samples.append(float(sector_weight))
            sector_distributions_by_repeat.append(dict(sector_dist))
            if requested_symmetry_mode in {"postselect_diag_v1", "projector_renorm_v1"}:
                observable_weight_map, retained_fraction = _postselected_weight_map_and_fraction(
                    base_weight_map,
                    n_qubits=int(num_qubits),
                    symmetry_cfg=symmetry_cfg,
                )
            else:
                observable_weight_map = dict(base_weight_map)
                retained_fraction = 1.0
            retained_fraction_samples.append(float(retained_fraction))
            doublon_total_samples.append(
                float(
                    _diagonal_expectation_from_weight_map(
                        observable_weight_map,
                        doublon_qop,
                        n_qubits=int(num_qubits),
                    )
                )
            )
            for site, observable in enumerate(site_doublon_qops):
                site_doublon_samples[int(site)].append(
                    float(
                        _diagonal_expectation_from_weight_map(
                            observable_weight_map,
                            observable,
                            n_qubits=int(num_qubits),
                        )
                    )
                )
            for site, observable in enumerate(site_charge_qops):
                site_charge_samples[int(site)].append(
                    float(
                        _diagonal_expectation_from_weight_map(
                            observable_weight_map,
                            observable,
                            n_qubits=int(num_qubits),
                        )
                    )
                )

        sector_estimate = _oracle_estimate_from_samples(sector_weight_samples, aggregate="mean")
        retained_estimate = _oracle_estimate_from_samples(
            retained_fraction_samples, aggregate="mean"
        )
        doublon_estimate = _oracle_estimate_from_samples(doublon_total_samples, aggregate="mean")
        sector_keys = {
            (int(sector_n_up_resolved), int(sector_n_dn_resolved)),
            *{
                (int(key[0]), int(key[1]))
                for dist in sector_distributions_by_repeat
                for key in dist.keys()
            },
        }
        weights_by_sector: list[dict[str, Any]] = []
        for n_up, n_dn in sorted(sector_keys):
            weight_estimate = _oracle_estimate_from_samples(
                [
                    float(dist.get((int(n_up), int(n_dn)), 0.0))
                    for dist in sector_distributions_by_repeat
                ],
                aggregate="mean",
            )
            weights_by_sector.append(
                {
                    "n_up": int(n_up),
                    "n_dn": int(n_dn),
                    "mean": float(weight_estimate.mean),
                    "std": float(weight_estimate.std),
                    "stderr": float(weight_estimate.stderr),
                }
            )
        site_doublon_estimates = [
            _oracle_estimate_from_samples(samples, aggregate="mean")
            for samples in site_doublon_samples
        ]
        site_charge_estimates = [
            _oracle_estimate_from_samples(samples, aggregate="mean")
            for samples in site_charge_samples
        ]
        repeat_count = int(len(views))
        readout_applied = mitigation_mode == "readout"
        unique_active = {
            tuple(int(q) for q in qubits) for qubits in active_qubits_seen if qubits is not None
        }
        estimator_form = {
            "off": "raw_diagonal_average",
            "verify_only": "raw_diagonal_average",
            "postselect_diag_v1": "postselected_diag_v1",
            "projector_renorm_v1": "projector_ratio_diag_v1",
        }[requested_symmetry_mode]
        summary = {
            "source": "raw_full_register_z_postprocessed_v1",
            "available": True,
            "evaluation_id": str(eval_id),
            "observable_family": observable_family,
            "num_qubits": int(num_qubits),
            "num_sites": int(num_sites_resolved),
            "ordering": str(ordering_resolved),
            "target_sector": {
                "n_up": int(sector_n_up_resolved),
                "n_dn": int(sector_n_dn_resolved),
            },
            "repeat_count": int(repeat_count),
            "expected_repeat_count": (
                None if expected_repeat_count is None else int(expected_repeat_count)
            ),
            "repeat_grid_complete": (
                True
                if expected_repeat_count is None
                else int(repeat_count) == int(expected_repeat_count)
            ),
            "pipeline": {
                "readout_mode": str(mitigation_mode),
                "symmetry_mode": str(requested_symmetry_mode),
                "applied_symmetry_mode": str(requested_symmetry_mode),
                "order": "readout_then_symmetry",
                "observable_scope": "full_register_diagonal_only",
                "estimator_form": str(estimator_form),
            },
            "sector_weight_mean": float(sector_estimate.mean),
            "sector_weight_std": float(sector_estimate.std),
            "sector_weight_stderr": float(sector_estimate.stderr),
            "sector_weight_samples": [float(x) for x in sector_estimate.raw_values],
            "retained_fraction_mean": float(retained_estimate.mean),
            "retained_fraction_std": float(retained_estimate.std),
            "retained_fraction_stderr": float(retained_estimate.stderr),
            "retained_fraction_samples": [
                float(x) for x in retained_estimate.raw_values
            ],
            "doublon_total_mean": float(doublon_estimate.mean),
            "doublon_total_std": float(doublon_estimate.std),
            "doublon_total_stderr": float(doublon_estimate.stderr),
            "doublon_total_samples": [float(x) for x in doublon_estimate.raw_values],
            "sector_distribution": {
                "weights_by_sector": list(weights_by_sector),
                "sorted_by": "n_up_then_n_dn",
            },
            "site_observables": {
                "site_index_order": "physical_site_0_to_n_minus_1",
                "doublon_by_site_mean": [
                    float(est.mean) for est in site_doublon_estimates
                ],
                "doublon_by_site_std": [float(est.std) for est in site_doublon_estimates],
                "doublon_by_site_stderr": [
                    float(est.stderr) for est in site_doublon_estimates
                ],
                "charge_by_site_mean": [float(est.mean) for est in site_charge_estimates],
                "charge_by_site_std": [float(est.std) for est in site_charge_estimates],
                "charge_by_site_stderr": [
                    float(est.stderr) for est in site_charge_estimates
                ],
            },
            "observable_span": {
                "kind": "full_register_z_basis_v1",
                "supports_sector_weight": True,
                "supports_doublon_total": True,
                "supports_postselected_diagonal_observables": True,
                "supports_projector_renorm_diagonal_observables": True,
                "supports_postselected_energy": False,
                "supports_projector_renorm_energy": False,
                "reason": "full_register_z_basis_only",
            },
        }
        return {
            "success": True,
            "available": True,
            "reason": None,
            "summary": summary,
            "readout_details": {
                "applied": bool(readout_applied),
                "strategy": (
                    None
                    if not readout_applied
                    else str(mitigation_cfg.get("local_readout_strategy") or "mthree")
                ),
                "active_physical_qubits": (
                    None if len(unique_active) != 1 else list(next(iter(unique_active)))
                ),
                "calibration_cache_size": int(len(calibrated_qubits)) if readout_applied else 0,
                "negative_mass_mean": (
                    None
                    if not negative_mass_samples
                    else float(np.mean(np.asarray(negative_mass_samples, dtype=float)))
                ),
                "negative_mass_samples": [float(x) for x in negative_mass_samples],
                "mitigation_overhead_mean": (
                    None
                    if not mitigation_overhead_samples
                    else float(np.mean(np.asarray(mitigation_overhead_samples, dtype=float)))
                ),
                "mitigation_overhead_samples": [
                    float(x) for x in mitigation_overhead_samples
                ],
            },
            "error_type": None,
            "error_message": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "available": False,
            "reason": "postprocessing_failed",
            "summary": None,
            "readout_details": None,
            "error_type": str(type(exc).__name__),
            "error_message": str(exc),
        }


def _load_raw_measurement_artifact_records(
    path: str,
    *,
    evaluation_id: str | None = None,
    observable_family: str | None = None,
) -> list[dict[str, Any]]:
    path_s = str(path).strip()
    if path_s == "":
        raise ValueError("raw artifact path must be non-empty.")
    rows: list[dict[str, Any]] = []
    open_fn: Callable[..., Any]
    open_kwargs: dict[str, Any] = {"encoding": "utf-8"}
    if path_s.endswith(".gz"):
        open_fn = gzip.open
        open_kwargs["mode"] = "rt"
    else:
        open_fn = open
        open_kwargs["mode"] = "r"
    with open_fn(path_s, **open_kwargs) as fh:
        for line_index, line in enumerate(fh, start=1):
            line_s = str(line).strip()
            if line_s == "":
                continue
            try:
                payload = json.loads(line_s)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed raw measurement artifact JSON at {path_s}:{line_index}."
                ) from exc
            row = _raw_record_view(payload)
            if evaluation_id not in {None, ""} and str(row["evaluation_id"]) != str(evaluation_id):
                continue
            if observable_family not in {None, ""} and str(row["observable_family"]) != str(observable_family):
                continue
            rows.append(row)
    return rows


def _summarize_hh_full_register_z_records(
    records: Sequence[RawMeasurementRecord | Mapping[str, Any]],
    *,
    num_sites: int | None = None,
    ordering: str | None = None,
    sector_n_up: int | None = None,
    sector_n_dn: int | None = None,
    expected_repeat_count: int | None = None,
) -> dict[str, Any]:
    if len(records) <= 0:
        raise ValueError("HH symmetry summary requires at least one raw record.")
    views = [_raw_record_view(record) for record in records]
    first = dict(views[0])
    semantic_tags = dict(first.get("semantic_tags", {}))
    eval_id = str(first.get("evaluation_id", ""))
    if eval_id == "":
        raise ValueError("HH symmetry summary requires a non-empty evaluation_id.")
    observable_family = first.get("observable_family", None)
    num_qubits = int(first.get("num_qubits", 0))
    if num_qubits <= 0:
        raise ValueError("HH symmetry summary requires num_qubits > 0.")

    def _resolve_int(explicit: int | None, tag_key: str) -> int | None:
        if explicit is not None:
            return int(explicit)
        raw = semantic_tags.get(tag_key, None)
        if raw in {None, ""}:
            return None
        return int(raw)

    num_sites_resolved = _resolve_int(num_sites, "symmetry_num_sites")
    sector_n_up_resolved = _resolve_int(sector_n_up, "symmetry_sector_n_up")
    sector_n_dn_resolved = _resolve_int(sector_n_dn, "symmetry_sector_n_dn")
    ordering_resolved = (
        str(ordering)
        if ordering not in {None, ""}
        else str(semantic_tags.get("symmetry_ordering", "blocked"))
    ).strip().lower() or "blocked"

    if num_sites_resolved is None:
        raise ValueError("HH symmetry summary requires num_sites metadata.")
    if sector_n_up_resolved is None or sector_n_dn_resolved is None:
        raise ValueError("HH symmetry summary requires target-sector metadata.")
    if int(num_qubits) < 2 * int(num_sites_resolved):
        raise ValueError(
            "HH symmetry summary requires num_qubits to cover the full fermion register."
        )

    expected_measured = tuple(range(int(num_qubits)))
    seen_repeat_indices: set[int] = set()
    for row in views:
        if str(row.get("evaluation_id", "")) != str(eval_id):
            raise ValueError("HH symmetry summary requires exactly one evaluation_id.")
        if int(row.get("num_qubits", 0)) != int(num_qubits):
            raise ValueError("HH symmetry summary requires a consistent num_qubits across records.")
        if row.get("observable_family", None) != observable_family:
            raise ValueError("HH symmetry summary requires a consistent observable_family.")
        if str(row.get("basis_label", "")).upper() != ("Z" * int(num_qubits)):
            raise ValueError("HH symmetry summary requires full-register all-Z basis records only.")
        measured_logical = tuple(int(x) for x in tuple(row.get("measured_logical_qubits", ())))
        if measured_logical != expected_measured:
            raise ValueError(
                "HH symmetry summary requires full-register measured_logical_qubits coverage."
            )
        repeat_index = int(row.get("repeat_index", 0))
        if repeat_index in seen_repeat_indices:
            raise ValueError("HH symmetry summary requires unique repeat_index values.")
        seen_repeat_indices.add(repeat_index)

    symmetry_cfg = {
        "mode": "verify_only",
        "num_sites": int(num_sites_resolved),
        "ordering": str(ordering_resolved),
        "sector_n_up": int(sector_n_up_resolved),
        "sector_n_dn": int(sector_n_dn_resolved),
    }
    doublon_qop = _hh_total_doublon_qop(
        num_qubits=int(num_qubits),
        num_sites=int(num_sites_resolved),
        ordering=str(ordering_resolved),
    )
    alpha_indices, beta_indices = _spin_orbital_index_sets(
        int(num_sites_resolved),
        str(ordering_resolved),
    )
    site_index_pairs = [
        (int(alpha_indices[site]), int(beta_indices[site]))
        for site in range(int(num_sites_resolved))
    ]
    site_doublon_qops = [
        _doublon_site_qop(int(num_qubits), int(up_idx), int(dn_idx))
        for up_idx, dn_idx in site_index_pairs
    ]
    site_charge_qops = [
        (
            _number_operator_qop(int(num_qubits), int(up_idx))
            + _number_operator_qop(int(num_qubits), int(dn_idx))
        ).simplify(atol=1e-12)
        for up_idx, dn_idx in site_index_pairs
    ]

    def _sector_distribution_from_counts(counts: Mapping[str, int]) -> dict[tuple[int, int], float]:
        total_shots = int(sum(int(v) for v in counts.values()))
        if total_shots <= 0:
            raise RuntimeError("HH symmetry summary requires strictly positive shots per repeat.")
        weights: dict[tuple[int, int], float] = {}
        for bitstr_raw, ct_raw in counts.items():
            bitstr = str(bitstr_raw).replace(" ", "")
            if len(bitstr) < int(num_qubits):
                bitstr = bitstr.zfill(int(num_qubits))
            n_up = int(sum(1 for idx in alpha_indices if bitstr[-1 - int(idx)] == "1"))
            n_dn = int(sum(1 for idx in beta_indices if bitstr[-1 - int(idx)] == "1"))
            key = (int(n_up), int(n_dn))
            weights[key] = float(weights.get(key, 0.0) + (float(int(ct_raw)) / float(total_shots)))
        return weights

    sector_weight_samples: list[float] = []
    doublon_total_samples: list[float] = []
    sector_distributions_by_repeat: list[dict[tuple[int, int], float]] = []
    site_doublon_samples: list[list[float]] = [[] for _ in range(int(num_sites_resolved))]
    site_charge_samples: list[list[float]] = [[] for _ in range(int(num_sites_resolved))]
    for row in sorted(views, key=lambda item: int(item["repeat_index"])):
        counts = {str(k): int(v) for k, v in dict(row["counts"]).items()}
        try:
            _kept_counts, sector_weight = _postselected_counts_and_fraction(
                counts,
                n_qubits=int(num_qubits),
                symmetry_cfg=symmetry_cfg,
            )
        except RuntimeError as exc:
            if "retained zero shots" in str(exc):
                sector_weight = 0.0
            else:
                raise
        sector_weight_samples.append(float(sector_weight))
        doublon_total_samples.append(float(_diagonal_expectation_from_counts(counts, doublon_qop)))
        sector_distributions_by_repeat.append(_sector_distribution_from_counts(counts))
        for site, observable in enumerate(site_doublon_qops):
            site_doublon_samples[int(site)].append(
                float(_diagonal_expectation_from_counts(counts, observable))
            )
        for site, observable in enumerate(site_charge_qops):
            site_charge_samples[int(site)].append(
                float(_diagonal_expectation_from_counts(counts, observable))
            )
    sector_estimate = _oracle_estimate_from_samples(sector_weight_samples, aggregate="mean")
    doublon_estimate = _oracle_estimate_from_samples(doublon_total_samples, aggregate="mean")
    sector_keys = {
        (int(sector_n_up_resolved), int(sector_n_dn_resolved)),
        *{
            (int(key[0]), int(key[1]))
            for dist in sector_distributions_by_repeat
            for key in dist.keys()
        },
    }
    weights_by_sector: list[dict[str, Any]] = []
    for n_up, n_dn in sorted(sector_keys):
        weight_estimate = _oracle_estimate_from_samples(
            [float(dist.get((int(n_up), int(n_dn)), 0.0)) for dist in sector_distributions_by_repeat],
            aggregate="mean",
        )
        weights_by_sector.append(
            {
                "n_up": int(n_up),
                "n_dn": int(n_dn),
                "mean": float(weight_estimate.mean),
                "std": float(weight_estimate.std),
                "stderr": float(weight_estimate.stderr),
            }
        )
    site_doublon_estimates = [
        _oracle_estimate_from_samples(samples, aggregate="mean")
        for samples in site_doublon_samples
    ]
    site_charge_estimates = [
        _oracle_estimate_from_samples(samples, aggregate="mean")
        for samples in site_charge_samples
    ]
    repeat_count = int(len(views))
    return {
        "source": "raw_all_z_full_register_v1",
        "available": True,
        "evaluation_id": str(eval_id),
        "observable_family": observable_family,
        "num_qubits": int(num_qubits),
        "num_sites": int(num_sites_resolved),
        "ordering": str(ordering_resolved),
        "target_sector": {
            "n_up": int(sector_n_up_resolved),
            "n_dn": int(sector_n_dn_resolved),
        },
        "repeat_count": int(repeat_count),
        "expected_repeat_count": (
            None if expected_repeat_count is None else int(expected_repeat_count)
        ),
        "repeat_grid_complete": (
            True if expected_repeat_count is None else int(repeat_count) == int(expected_repeat_count)
        ),
        "sector_weight_mean": float(sector_estimate.mean),
        "sector_weight_std": float(sector_estimate.std),
        "sector_weight_stderr": float(sector_estimate.stderr),
        "sector_weight_samples": [float(x) for x in sector_estimate.raw_values],
        "doublon_total_mean": float(doublon_estimate.mean),
        "doublon_total_std": float(doublon_estimate.std),
        "doublon_total_stderr": float(doublon_estimate.stderr),
        "doublon_total_samples": [float(x) for x in doublon_estimate.raw_values],
        "sector_distribution": {
            "weights_by_sector": list(weights_by_sector),
            "sorted_by": "n_up_then_n_dn",
        },
        "site_observables": {
            "site_index_order": "physical_site_0_to_n_minus_1",
            "doublon_by_site_mean": [float(est.mean) for est in site_doublon_estimates],
            "doublon_by_site_std": [float(est.std) for est in site_doublon_estimates],
            "doublon_by_site_stderr": [float(est.stderr) for est in site_doublon_estimates],
            "charge_by_site_mean": [float(est.mean) for est in site_charge_estimates],
            "charge_by_site_std": [float(est.std) for est in site_charge_estimates],
            "charge_by_site_stderr": [float(est.stderr) for est in site_charge_estimates],
        },
        "observable_span": {
            "kind": "full_register_z_basis_v1",
            "supports_sector_weight": True,
            "supports_doublon_total": True,
            "supports_postselected_energy": False,
            "supports_projector_renorm_energy": False,
            "reason": "full_register_z_basis_only",
        },
    }


def _summarize_hh_full_register_z_artifact(
    path: str,
    *,
    evaluation_id: str | None = None,
    observable_family: str | None = None,
    num_sites: int | None = None,
    ordering: str | None = None,
    sector_n_up: int | None = None,
    sector_n_dn: int | None = None,
    expected_repeat_count: int | None = None,
) -> dict[str, Any]:
    rows = _load_raw_measurement_artifact_records(
        path,
        evaluation_id=evaluation_id,
        observable_family=observable_family,
    )
    if len(rows) <= 0:
        raise ValueError("No raw measurement artifact rows matched the requested HH symmetry filter.")
    eval_ids = {str(row.get("evaluation_id", "")) for row in rows}
    if len(eval_ids) != 1:
        raise ValueError(
            "HH symmetry artifact summary requires exactly one evaluation_id after filtering."
        )
    summary = _summarize_hh_full_register_z_records(
        rows,
        num_sites=num_sites,
        ordering=ordering,
        sector_n_up=sector_n_up,
        sector_n_dn=sector_n_dn,
        expected_repeat_count=expected_repeat_count,
    )
    summary["raw_artifact_path"] = str(path)
    return summary


def _bootstrap_hh_full_register_z_records(
    records: Sequence[RawMeasurementRecord | Mapping[str, Any]],
    *,
    num_sites: int | None = None,
    ordering: str | None = None,
    sector_n_up: int | None = None,
    sector_n_dn: int | None = None,
    expected_repeat_count: int | None = None,
    bootstrap_repetitions: int = 128,
    confidence_level: float = 0.95,
    seed: int = 7,
) -> dict[str, Any]:
    if int(bootstrap_repetitions) <= 0:
        raise ValueError("bootstrap_repetitions must be positive.")
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError("confidence_level must lie strictly between 0 and 1.")
    base_summary = _summarize_hh_full_register_z_records(
        records,
        num_sites=num_sites,
        ordering=ordering,
        sector_n_up=sector_n_up,
        sector_n_dn=sector_n_dn,
        expected_repeat_count=expected_repeat_count,
    )
    views = [_raw_record_view(record) for record in records]
    rng = np.random.default_rng(int(seed))
    alpha_tail = 100.0 * (1.0 - float(confidence_level)) / 2.0
    lower_pct = float(alpha_tail)
    upper_pct = float(100.0 - alpha_tail)

    def _resample_counts(counts: Mapping[str, int]) -> dict[str, int]:
        bitstrings = [str(bitstr) for bitstr in counts.keys()]
        shots = int(sum(int(v) for v in counts.values()))
        if shots <= 0:
            raise RuntimeError("HH symmetry bootstrap requires strictly positive shots.")
        probs = np.asarray([float(int(counts[bitstr])) for bitstr in bitstrings], dtype=float)
        probs /= float(np.sum(probs))
        sampled = rng.multinomial(int(shots), probs)
        return {
            str(bitstr): int(count)
            for bitstr, count in zip(bitstrings, sampled)
            if int(count) > 0
        }

    bootstrap_sector_weight: list[float] = []
    bootstrap_doublon_total: list[float] = []
    sector_keys = [
        (int(row["n_up"]), int(row["n_dn"]))
        for row in list(base_summary.get("sector_distribution", {}).get("weights_by_sector", []))
    ]
    bootstrap_sector_distribution: dict[tuple[int, int], list[float]] = {
        key: [] for key in sector_keys
    }
    num_sites_resolved = int(base_summary["num_sites"])
    bootstrap_doublon_by_site: list[list[float]] = [[] for _ in range(num_sites_resolved)]
    bootstrap_charge_by_site: list[list[float]] = [[] for _ in range(num_sites_resolved)]

    for _ in range(int(bootstrap_repetitions)):
        resampled_rows: list[dict[str, Any]] = []
        for view in views:
            row = dict(view)
            row["counts"] = _resample_counts(row["counts"])
            row["shots_completed"] = int(sum(int(v) for v in row["counts"].values()))
            resampled_rows.append(row)
        resampled_summary = _summarize_hh_full_register_z_records(
            resampled_rows,
            num_sites=num_sites,
            ordering=ordering,
            sector_n_up=sector_n_up,
            sector_n_dn=sector_n_dn,
            expected_repeat_count=expected_repeat_count,
        )
        bootstrap_sector_weight.append(float(resampled_summary["sector_weight_mean"]))
        bootstrap_doublon_total.append(float(resampled_summary["doublon_total_mean"]))
        resampled_sector_map = {
            (int(row["n_up"]), int(row["n_dn"])): float(row["mean"])
            for row in list(resampled_summary.get("sector_distribution", {}).get("weights_by_sector", []))
        }
        for key in sector_keys:
            bootstrap_sector_distribution[key].append(float(resampled_sector_map.get(key, 0.0)))
        resampled_site_observables = dict(resampled_summary.get("site_observables", {}))
        doublon_means = list(resampled_site_observables.get("doublon_by_site_mean", []))
        charge_means = list(resampled_site_observables.get("charge_by_site_mean", []))
        for site in range(num_sites_resolved):
            bootstrap_doublon_by_site[site].append(float(doublon_means[site]))
            bootstrap_charge_by_site[site].append(float(charge_means[site]))

    def _metric(point_estimate: float, samples: Sequence[float]) -> dict[str, Any]:
        arr = np.asarray(list(samples), dtype=float)
        return {
            "point_estimate": float(point_estimate),
            "bootstrap_mean": float(np.mean(arr)),
            "bootstrap_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "ci_lower": float(np.percentile(arr, lower_pct)),
            "ci_upper": float(np.percentile(arr, upper_pct)),
        }

    return {
        "source": "hh_full_register_z_bootstrap_v1",
        "available": True,
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "confidence_level": float(confidence_level),
        "evaluation_id": str(base_summary["evaluation_id"]),
        "observable_family": base_summary.get("observable_family", None),
        "raw_metric_summary": {
            "sector_weight_mean": float(base_summary["sector_weight_mean"]),
            "doublon_total_mean": float(base_summary["doublon_total_mean"]),
        },
        "metrics": {
            "sector_weight": _metric(
                float(base_summary["sector_weight_mean"]),
                bootstrap_sector_weight,
            ),
            "doublon_total": _metric(
                float(base_summary["doublon_total_mean"]),
                bootstrap_doublon_total,
            ),
            "sector_distribution": [
                {
                    "n_up": int(n_up),
                    "n_dn": int(n_dn),
                    **_metric(
                        float(row["mean"]),
                        bootstrap_sector_distribution[(int(n_up), int(n_dn))],
                    ),
                }
                for row, (n_up, n_dn) in zip(
                    list(base_summary.get("sector_distribution", {}).get("weights_by_sector", [])),
                    sector_keys,
                )
            ],
            "doublon_by_site": [
                {
                    "site": int(site),
                    **_metric(
                        float(base_summary["site_observables"]["doublon_by_site_mean"][site]),
                        bootstrap_doublon_by_site[site],
                    ),
                }
                for site in range(num_sites_resolved)
            ],
            "charge_by_site": [
                {
                    "site": int(site),
                    **_metric(
                        float(base_summary["site_observables"]["charge_by_site_mean"][site]),
                        bootstrap_charge_by_site[site],
                    ),
                }
                for site in range(num_sites_resolved)
            ],
        },
    }


def _bootstrap_hh_full_register_z_artifact(
    path: str,
    *,
    evaluation_id: str | None = None,
    observable_family: str | None = None,
    num_sites: int | None = None,
    ordering: str | None = None,
    sector_n_up: int | None = None,
    sector_n_dn: int | None = None,
    expected_repeat_count: int | None = None,
    bootstrap_repetitions: int = 128,
    confidence_level: float = 0.95,
    seed: int = 7,
) -> dict[str, Any]:
    rows = _load_raw_measurement_artifact_records(
        path,
        evaluation_id=evaluation_id,
        observable_family=observable_family,
    )
    if len(rows) <= 0:
        raise ValueError("No raw measurement artifact rows matched the requested HH symmetry bootstrap filter.")
    summary = _bootstrap_hh_full_register_z_records(
        rows,
        num_sites=num_sites,
        ordering=ordering,
        sector_n_up=sector_n_up,
        sector_n_dn=sector_n_dn,
        expected_repeat_count=expected_repeat_count,
        bootstrap_repetitions=int(bootstrap_repetitions),
        confidence_level=float(confidence_level),
        seed=int(seed),
    )
    summary["raw_artifact_path"] = str(path)
    return summary


def _run_sampler_fallback_job(
    sampler: Any,
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
) -> float:
    total = 0.0 + 0.0j
    n = int(observable.num_qubits)
    for label, coeff in observable.to_list():
        lbl = str(label).upper()
        meas_qc = _term_measurement_circuit(circuit, lbl)
        job = sampler.run([meas_qc])
        result = job.result()
        counts = result[0].join_data().get_counts()
        exp_lbl = _pauli_expectation_from_counts(counts, lbl, n)
        total += complex(coeff) * complex(exp_lbl, 0.0)
    return float(np.real(total))


class RawMeasurementOracle:
    """Raw-shot-first grouped measurement oracle for runtime/backend_scheduled execution."""

    def __init__(self, config: OracleConfig):
        transpile_seed_raw = getattr(config, "seed_transpiler", None)
        transpile_seed = None if transpile_seed_raw is None else int(transpile_seed_raw)
        transpile_opt_level = int(getattr(config, "transpile_optimization_level", 1))
        if transpile_opt_level < 0 or transpile_opt_level > 3:
            raise ValueError("transpile_optimization_level must be between 0 and 3 inclusive.")
        requested_execution_surface = str(
            getattr(config, "execution_surface", "raw_measurement_v1")
        ).strip().lower()
        if requested_execution_surface in {"", "expectation_v1"}:
            requested_execution_surface = "raw_measurement_v1"
        self.config = OracleConfig(
            noise_mode=str(config.noise_mode).strip().lower(),
            shots=int(config.shots),
            seed=int(config.seed),
            seed_transpiler=transpile_seed,
            transpile_optimization_level=int(transpile_opt_level),
            oracle_repeats=max(1, int(config.oracle_repeats)),
            oracle_aggregate=str(config.oracle_aggregate).strip().lower(),
            backend_name=(None if config.backend_name is None else str(config.backend_name)),
            use_fake_backend=bool(config.use_fake_backend),
            approximation=bool(config.approximation),
            abelian_grouping=bool(config.abelian_grouping),
            allow_aer_fallback=bool(config.allow_aer_fallback),
            aer_fallback_mode=str(config.aer_fallback_mode).strip().lower(),
            omp_shm_workaround=bool(config.omp_shm_workaround),
            mitigation=normalize_mitigation_config(getattr(config, "mitigation", "none")),
            symmetry_mitigation=normalize_symmetry_mitigation_config(
                getattr(config, "symmetry_mitigation", "off")
            ),
            runtime_profile=normalize_runtime_estimator_profile_config(
                getattr(config, "runtime_profile", "legacy_runtime_v0")
            ),
            runtime_session=normalize_runtime_session_policy_config(
                getattr(config, "runtime_session", "prefer_session")
            ),
            execution_surface=str(requested_execution_surface),
            raw_transport=str(getattr(config, "raw_transport", "auto")).strip().lower() or "auto",
            raw_store_memory=bool(getattr(config, "raw_store_memory", False)),
            raw_grouping_mode=str(
                getattr(config, "raw_grouping_mode", "qwc_basis_cover_reuse")
            ).strip().lower()
            or "qwc_basis_cover_reuse",
            raw_artifact_path=(
                None
                if getattr(config, "raw_artifact_path", None) in {None, ""}
                else str(getattr(config, "raw_artifact_path"))
            ),
        )
        if self.config.oracle_aggregate not in {"mean", "median"}:
            raise ValueError(
                f"Unsupported oracle_aggregate={self.config.oracle_aggregate}; use mean or median."
            )
        if self.config.noise_mode not in {"runtime", "backend_scheduled"}:
            raise ValueError("RawMeasurementOracle supports only runtime/backend_scheduled noise modes.")
        if str(self.config.execution_surface) != "raw_measurement_v1":
            raise ValueError("RawMeasurementOracle requires execution_surface='raw_measurement_v1'.")
        if str(self.config.raw_grouping_mode) != "qwc_basis_cover_reuse":
            raise ValueError(
                "RawMeasurementOracle currently supports only raw_grouping_mode='qwc_basis_cover_reuse'."
            )
        mitigation_cfg = normalize_mitigation_config(getattr(self.config, "mitigation", "none"))
        if str(mitigation_cfg.get("mode", "none")) != "none":
            raise ValueError("RawMeasurementOracle phase-1 acquisition requires mitigation mode 'none'.")
        symmetry_cfg = normalize_symmetry_mitigation_config(
            getattr(self.config, "symmetry_mitigation", "off")
        )
        if str(symmetry_cfg.get("mode", "off")) not in {"off", "verify_only"}:
            raise ValueError(
                "RawMeasurementOracle phase-1 supports symmetry_mitigation only in {'off','verify_only'}."
            )
        runtime_profile_cfg = normalize_runtime_estimator_profile_config(
            getattr(self.config, "runtime_profile", "legacy_runtime_v0")
        )
        if (
            self.config.noise_mode == "runtime"
            and str(runtime_profile_cfg.get("name", "legacy_runtime_v0")) != "legacy_runtime_v0"
        ):
            raise ValueError(
                "RawMeasurementOracle phase-1 runtime mode requires runtime_profile='legacy_runtime_v0'."
            )
        if self.config.noise_mode == "backend_scheduled" and not bool(self.config.use_fake_backend):
            raise ValueError("RawMeasurementOracle backend_scheduled mode requires use_fake_backend=True.")
        self._backend_target = None
        self._runtime_mode = None
        self._session = None
        self._runtime_sampler = None
        self._runtime_sampler_configured_details: dict[str, Any] | None = None
        self._compiled_group_template_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._memory_records: list[RawMeasurementRecord] = []
        self._evaluation_counter = 0
        self._closed = False
        if self.config.noise_mode == "backend_scheduled":
            backend_obj, _backend_name, _using_fake = _resolve_noise_backend(self.config)
            self._backend_target = backend_obj
            self._runtime_mode = backend_obj
        else:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService, Session
            except Exception as exc:
                raise RuntimeError(
                    "RawMeasurementOracle runtime mode requires qiskit-ibm-runtime."
                ) from exc
            if self.config.backend_name is None:
                raise RuntimeError("RawMeasurementOracle runtime mode requires backend_name.")
            service = QiskitRuntimeService()
            backend_obj = service.backend(str(self.config.backend_name))
            runtime_session_cfg = normalize_runtime_session_policy_config(
                getattr(self.config, "runtime_session", "prefer_session")
            )
            session_policy = str(runtime_session_cfg.get("mode", "prefer_session"))
            session = None
            runtime_mode = backend_obj
            if session_policy != "backend_only":
                try:
                    try:
                        session = Session(service=service, backend=backend_obj)
                    except TypeError:
                        session = Session(backend=backend_obj)
                    runtime_mode = session
                except Exception as session_exc:
                    if session_policy == "require_session":
                        raise RuntimeError(
                            "Runtime session initialization failed while session batching was required. "
                            f"Details: {type(session_exc).__name__}: {session_exc}"
                        ) from session_exc
            self._backend_target = backend_obj
            self._runtime_mode = runtime_mode
            self._session = session
        self.backend_snapshot = dict(_snapshot_backend_target_shared(self._backend_target))
        self.transport = _resolve_raw_transport(self.config, self._backend_target)

    def close(self) -> None:
        if self._closed:
            return
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
        self._closed = True

    def __enter__(self) -> "RawMeasurementOracle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def memory_records(self) -> tuple[RawMeasurementRecord, ...]:
        return tuple(self._memory_records)

    @property
    def backend_target(self) -> Any | None:
        return self._backend_target

    def _get_runtime_sampler(self) -> Any:
        if self.transport != "sampler_v2":
            raise RuntimeError("runtime sampler is unavailable when raw transport is not sampler_v2.")
        if self._runtime_sampler is not None:
            return self._runtime_sampler
        try:
            from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2
        except Exception as exc:
            raise RuntimeError(
                "runtime raw measurement requires qiskit-ibm-runtime SamplerV2."
            ) from exc
        sampler = RuntimeSamplerV2(mode=self._runtime_mode)
        runtime_profile_cfg = normalize_runtime_estimator_profile_config(
            getattr(self.config, "runtime_profile", "legacy_runtime_v0")
        )
        self._runtime_sampler_configured_details = _configure_runtime_sampler_options(
            sampler,
            cfg=self.config,
            runtime_profile_cfg=runtime_profile_cfg,
        )
        self._runtime_sampler = sampler
        return sampler

    def _template_cache_key(self, plan: Any, basis_label: str) -> tuple[Any, ...]:
        transpile_seed = int(
            self.config.seed if self.config.seed_transpiler is None else self.config.seed_transpiler
        )
        return (
            str(getattr(plan, "plan_digest")),
            str(basis_label).upper(),
            str(self.transport),
            _hash_payload(self.backend_snapshot),
            int(transpile_seed),
            int(self.config.transpile_optimization_level),
        )

    def _compile_signature(self, compiled_payload: Mapping[str, Any]) -> dict[str, Any]:
        compiled = compiled_payload["compiled"]
        logical_to_physical = tuple(int(x) for x in compiled_payload.get("logical_to_physical", ()))
        return {
            "transpile_seed": (
                None if self.config.seed_transpiler is None else int(self.config.seed_transpiler)
            ),
            "transpile_optimization_level": int(self.config.transpile_optimization_level),
            "layout_physical_qubits": [int(x) for x in logical_to_physical],
            "compiled_num_qubits": int(compiled_payload.get("compiled_num_qubits", compiled.num_qubits)),
            **_compiled_metrics_payload(compiled),
        }

    def _get_group_template(self, plan: Any, basis_label: str) -> dict[str, Any]:
        cache_key = self._template_cache_key(plan, basis_label)
        cached = self._compiled_group_template_cache.get(cache_key)
        if cached is not None:
            return cached
        measured, active_logical = _sparse_term_measurement_circuit(plan.circuit, str(basis_label).upper())
        transpile_seed = int(
            self.config.seed if self.config.seed_transpiler is None else self.config.seed_transpiler
        )
        compiled_payload = compile_circuit_for_local_backend(
            measured,
            self._backend_target,
            seed_transpiler=transpile_seed,
            optimization_level=int(self.config.transpile_optimization_level),
        )
        logical_to_physical = tuple(int(x) for x in compiled_payload["logical_to_physical"])
        measured_physical = tuple(int(logical_to_physical[q]) for q in active_logical)
        physical_to_logical = {str(int(p)): int(idx) for idx, p in enumerate(logical_to_physical)}
        cached = {
            "compiled": compiled_payload["compiled"],
            "basis_label": str(basis_label).upper(),
            "measured_logical_qubits": tuple(int(q) for q in active_logical),
            "measured_physical_qubits": tuple(int(q) for q in measured_physical),
            "logical_to_physical": tuple(int(x) for x in logical_to_physical),
            "physical_to_logical": dict(physical_to_logical),
            "compile_signature": self._compile_signature(compiled_payload),
        }
        self._compiled_group_template_cache[cache_key] = cached
        return cached

    def _bind_group_template(
        self,
        template: Mapping[str, Any],
        plan: Any,
        theta_runtime: np.ndarray | Sequence[float],
    ) -> QuantumCircuit:
        theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
        if int(theta_arr.size) != int(getattr(plan.layout, "runtime_parameter_count")):
            raise ValueError(
                f"theta_runtime length mismatch: got {theta_arr.size}, expected {plan.layout.runtime_parameter_count}."
            )
        value_by_name = {
            str(getattr(param, "name", str(param))): float(theta_arr[idx])
            for idx, param in enumerate(tuple(getattr(plan, "parameters")))
        }
        compiled = template["compiled"]
        assignments = {}
        for param in tuple(getattr(compiled, "parameters", ())):
            name = str(getattr(param, "name", str(param)))
            if name not in value_by_name:
                raise RuntimeError(
                    f"Compiled group template parameter {name!r} is missing from the ansatz plan."
                )
            assignments[param] = float(value_by_name[name])
        return compiled.assign_parameters(assignments, inplace=False)

    def measure_observable(
        self,
        *,
        plan: Any,
        theta_runtime: np.ndarray | Sequence[float],
        observable: SparsePauliOp,
        observable_family: str,
        semantic_tags: Mapping[str, Any] | None = None,
        evaluation_id: str | None = None,
        runtime_job_observer: Callable[[dict[str, Any]], None] | None = None,
        runtime_trace_context: Mapping[str, Any] | None = None,
    ) -> RawExecutionBundle:
        if self._closed:
            raise RuntimeError("RawMeasurementOracle is closed.")
        if not hasattr(plan, "circuit") or not hasattr(plan, "parameters") or not hasattr(plan, "layout"):
            raise TypeError("plan must provide circuit, parameters, and layout attributes.")
        theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
        if int(theta_arr.size) != int(getattr(plan.layout, "runtime_parameter_count")):
            raise ValueError(
                f"theta_runtime length mismatch: got {theta_arr.size}, expected {plan.layout.runtime_parameter_count}."
            )
        theta_digest = _hash_payload(
            {
                "schema_version": "raw_theta_v1",
                "values": [float(x) for x in theta_arr.tolist()],
            }
        )
        self._evaluation_counter += 1
        eval_id = (
            str(evaluation_id).strip()
            if evaluation_id not in {None, ""}
            else (
                "raw_eval_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_"
                f"pid{os.getpid()}_"
                f"n{self._evaluation_counter:06d}_"
                f"{uuid.uuid4().hex[:12]}"
            )
        )
        tags_base = dict(runtime_trace_context or {})
        tags_base.update({str(k): v for k, v in dict(semantic_tags or {}).items()})
        observable_terms = [
            (str(label).upper(), complex(coeff))
            for label, coeff in observable.to_list()
            if any(ch != "I" for ch in str(label).upper())
        ]
        groups = _group_observable_terms_by_qwc_basis(observable_terms)
        records: list[RawMeasurementRecord] = []
        compile_signatures_by_basis: dict[str, dict[str, Any]] = {}
        sampler = self._get_runtime_sampler() if self.transport == "sampler_v2" else None
        for group_index, group in enumerate(groups):
            basis_label = str(group["basis_label"]).upper()
            template = self._get_group_template(plan, basis_label)
            compile_signatures_by_basis[basis_label] = dict(template["compile_signature"])
            group_terms = tuple(
                {
                    "label": str(term_label),
                    "coeff_re": float(np.real(coeff)),
                    "coeff_im": float(np.imag(coeff)),
                }
                for term_label, coeff in group["terms"]
            )
            for repeat_idx in range(int(self.config.oracle_repeats)):
                bound_circuit = self._bind_group_template(template, plan, theta_arr)
                job_context = {
                    **tags_base,
                    "evaluation_id": str(eval_id),
                    "observable_family": str(observable_family),
                    "basis_label": str(basis_label),
                    "group_index": int(group_index),
                }
                execution = _run_raw_transport_job(
                    transport=self.transport,
                    sampler=sampler,
                    backend_target=self._backend_target,
                    circuit=bound_circuit,
                    shots=int(self.config.shots),
                    repeat_index=int(repeat_idx),
                    seed=int(self.config.seed),
                    job_observer=runtime_job_observer,
                    job_context=job_context,
                )
                counts = {str(bitstr): int(ct) for bitstr, ct in dict(execution.counts).items()}
                shots_completed = int(sum(int(v) for v in counts.values()))
                if shots_completed <= 0:
                    raise RuntimeError("Raw measurement execution returned zero shots.")
                record_id = _hash_payload(
                    {
                        "evaluation_id": str(eval_id),
                        "basis_label": str(basis_label),
                        "group_index": int(group_index),
                        "repeat_index": int(repeat_idx),
                        "plan_digest": str(getattr(plan, "plan_digest")),
                        "theta_digest": str(theta_digest),
                        "counts": dict(counts),
                    }
                )
                record = RawMeasurementRecord(
                    schema_version="raw_measurement_record_v1",
                    record_id=str(record_id),
                    evaluation_id=str(eval_id),
                    execution_surface="raw_measurement_v1",
                    observable_family=str(observable_family),
                    semantic_tags={
                        **tags_base,
                        "basis_label": str(basis_label),
                        "group_index": int(group_index),
                    },
                    group_index=int(group_index),
                    basis_label=str(basis_label),
                    group_terms=group_terms,
                    plan_digest=str(getattr(plan, "plan_digest")),
                    structure_digest=str(getattr(plan, "structure_digest")),
                    reference_state_digest=getattr(plan, "reference_state_digest", None),
                    theta_runtime=tuple(float(x) for x in theta_arr.tolist()),
                    theta_digest=str(theta_digest),
                    logical_parameter_count=int(getattr(plan.layout, "logical_parameter_count")),
                    runtime_parameter_count=int(getattr(plan.layout, "runtime_parameter_count")),
                    num_qubits=int(getattr(plan, "nq")),
                    measured_logical_qubits=tuple(int(q) for q in template["measured_logical_qubits"]),
                    measured_physical_qubits=(
                        tuple(int(q) for q in template["measured_physical_qubits"])
                        if template.get("measured_physical_qubits", None) is not None
                        else None
                    ),
                    logical_to_physical=tuple(int(q) for q in template["logical_to_physical"]),
                    physical_to_logical=dict(template["physical_to_logical"]),
                    compile_signature=dict(template["compile_signature"]),
                    backend_snapshot=dict(self.backend_snapshot),
                    transport=str(self.transport),
                    call_path=str(execution.used_call_path),
                    job_records=tuple(asdict(rec) for rec in execution.job_records),
                    repeat_index=int(repeat_idx),
                    shots_requested=int(self.config.shots),
                    shots_completed=int(shots_completed),
                    counts=dict(counts),
                    requested_mitigation=dict(
                        normalize_mitigation_config(getattr(self.config, "mitigation", "none"))
                    ),
                    requested_symmetry_mitigation=dict(
                        normalize_symmetry_mitigation_config(
                            getattr(self.config, "symmetry_mitigation", "off")
                        )
                    ),
                    requested_runtime_profile=dict(
                        normalize_runtime_estimator_profile_config(
                            getattr(self.config, "runtime_profile", "legacy_runtime_v0")
                        )
                    ),
                    requested_runtime_session=dict(
                        normalize_runtime_session_policy_config(
                            getattr(self.config, "runtime_session", "prefer_session")
                        )
                    ),
                    parent_record_ids=tuple(),
                    emitted_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
                if bool(self.config.raw_store_memory):
                    self._memory_records.append(record)
                if self.config.raw_artifact_path not in {None, ""}:
                    _write_raw_measurement_record(str(self.config.raw_artifact_path), record)
                records.append(record)
        estimate = _reduce_grouped_counts_to_observable(
            observable,
            records,
            aggregate=str(self.config.oracle_aggregate),
            expected_repeat_count=int(self.config.oracle_repeats),
        )
        return RawExecutionBundle(
            schema_version="raw_execution_bundle_v1",
            evaluation_id=str(eval_id),
            execution_surface="raw_measurement_v1",
            observable_family=str(observable_family),
            transport=str(self.transport),
            estimate=estimate,
            records=tuple(records),
            raw_artifact_path=(
                None if self.config.raw_artifact_path in {None, ""} else str(self.config.raw_artifact_path)
            ),
            plan_digest=str(getattr(plan, "plan_digest")),
            structure_digest=str(getattr(plan, "structure_digest")),
            reference_state_digest=getattr(plan, "reference_state_digest", None),
            compile_signatures_by_basis=dict(compile_signatures_by_basis),
            backend_snapshot=dict(self.backend_snapshot),
        )


class ExpectationOracle:
    """Shared expectation-value oracle for ideal/noisy/runtime execution."""

    def __init__(self, config: OracleConfig):
        transpile_seed_raw = getattr(config, "seed_transpiler", None)
        transpile_seed = None if transpile_seed_raw is None else int(transpile_seed_raw)
        transpile_opt_level = int(getattr(config, "transpile_optimization_level", 1))
        if transpile_opt_level < 0 or transpile_opt_level > 3:
            raise ValueError("transpile_optimization_level must be between 0 and 3 inclusive.")
        self.config = OracleConfig(
            noise_mode=str(config.noise_mode).strip().lower(),
            shots=int(config.shots),
            seed=int(config.seed),
            seed_transpiler=transpile_seed,
            transpile_optimization_level=int(transpile_opt_level),
            oracle_repeats=max(1, int(config.oracle_repeats)),
            oracle_aggregate=str(config.oracle_aggregate).strip().lower(),
            backend_name=(None if config.backend_name is None else str(config.backend_name)),
            use_fake_backend=bool(config.use_fake_backend),
            approximation=bool(config.approximation),
            abelian_grouping=bool(config.abelian_grouping),
            allow_aer_fallback=bool(config.allow_aer_fallback),
            aer_fallback_mode=str(config.aer_fallback_mode).strip().lower(),
            omp_shm_workaround=bool(config.omp_shm_workaround),
            mitigation=normalize_mitigation_config(getattr(config, "mitigation", "none")),
            symmetry_mitigation=normalize_symmetry_mitigation_config(
                getattr(config, "symmetry_mitigation", "off")
            ),
            runtime_profile=normalize_runtime_estimator_profile_config(
                getattr(config, "runtime_profile", "legacy_runtime_v0")
            ),
            runtime_session=normalize_runtime_session_policy_config(
                getattr(config, "runtime_session", "prefer_session")
            ),
            execution_surface=str(
                getattr(config, "execution_surface", "expectation_v1")
            ).strip().lower()
            or "expectation_v1",
            raw_transport=str(getattr(config, "raw_transport", "auto")).strip().lower() or "auto",
            raw_store_memory=bool(getattr(config, "raw_store_memory", False)),
            raw_grouping_mode=str(
                getattr(config, "raw_grouping_mode", "qwc_basis_cover_reuse")
            ).strip().lower()
            or "qwc_basis_cover_reuse",
            raw_artifact_path=(
                None
                if getattr(config, "raw_artifact_path", None) in {None, ""}
                else str(getattr(config, "raw_artifact_path"))
            ),
        )
        self._backend_target = None
        self._compiled_base_cache: dict[int, dict[str, Any]] = {}
        self._compiled_base_cache_hit_logged: set[int] = set()
        self._backend_scheduled_repeat_base_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self._backend_scheduled_attribution_targets: dict[str, dict[str, Any]] = {}
        self._backend_scheduled_noise_model = None
        self._backend_scheduled_local_dd_retry_target = None
        self._backend_scheduled_local_dd_retry_details: dict[str, Any] | None = None
        self._backend_scheduled_local_dd_retry_engaged = False
        self._runtime_sampler = None
        self._runtime_sampler_configured_details: dict[str, Any] | None = None
        self._runtime_measured_isa_cache: dict[tuple[int, str], dict[str, Any]] = {}
        self._mthree_module = None
        self._mthree_mitigator = None
        self._mthree_calibrated_qubits: set[tuple[int, ...]] = set()
        if self.config.oracle_aggregate not in {"mean", "median"}:
            raise ValueError(
                f"Unsupported oracle_aggregate={self.config.oracle_aggregate}; use mean or median."
            )
        if self.config.aer_fallback_mode not in {"sampler_shots"}:
            raise ValueError(
                f"Unsupported aer_fallback_mode={self.config.aer_fallback_mode}; use sampler_shots."
            )
        mitigation_cfg = normalize_mitigation_config(getattr(self.config, "mitigation", "none"))
        symmetry_cfg = normalize_symmetry_mitigation_config(
            getattr(self.config, "symmetry_mitigation", "off")
        )
        if self.config.noise_mode == "backend_scheduled":
            if not bool(self.config.use_fake_backend):
                raise ValueError("backend_scheduled requires use_fake_backend=True.")
            if str(mitigation_cfg.get("mode", "none")) not in {"none", "readout"}:
                raise ValueError(
                    "backend_scheduled currently supports only mitigation modes 'none' or 'readout'."
                )
            dd_sequence = mitigation_cfg.get("dd_sequence", None)
            if dd_sequence not in {None, "", "none"}:
                dd_sequence_norm = str(dd_sequence).strip().upper()
                if str(mitigation_cfg.get("mode", "none")) != "readout":
                    raise ValueError(
                        "backend_scheduled local DD requires mitigation mode 'readout'."
                    )
                if dd_sequence_norm != "XPXM":
                    raise ValueError(
                        "backend_scheduled local DD currently supports only dd_sequence='XpXm'."
                    )
                if bool(mitigation_cfg.get("local_gate_twirling", False)):
                    raise ValueError(
                        "backend_scheduled local DD is not combinable with local gate twirling."
                    )
            if str(mitigation_cfg.get("mode", "none")) == "readout":
                if str(symmetry_cfg.get("mode", "off")) not in {"off", "verify_only"}:
                    raise ValueError(
                        "readout mitigation is not combinable with active symmetry mitigation in backend_scheduled mode."
                    )
                strategy = str(mitigation_cfg.get("local_readout_strategy") or "mthree")
                if strategy not in _LOCAL_READOUT_STRATEGIES:
                    raise ValueError(
                        f"Unsupported backend_scheduled readout strategy {strategy!r}; "
                        f"expected one of {sorted(_LOCAL_READOUT_STRATEGIES)}."
                    )
                mitigation_cfg["local_readout_strategy"] = str(strategy)
                self.config = OracleConfig(
                    **{**self.config.__dict__, "mitigation": dict(mitigation_cfg)}
                )
                self._mthree_module = _resolve_mthree()
            try:
                import qiskit_aer  # noqa: F401
            except Exception as exc:  # pragma: no cover - import error path
                raise RuntimeError(
                    "backend_scheduled requires qiskit-aer so fake backends execute with a noise model."
                ) from exc

        self._sampler_fallback = None
        self._fallback_reason = ""
        self._estimator = None
        self._session = None
        self.backend_info = NoiseBackendInfo(
            noise_mode=str(self.config.noise_mode),
            estimator_kind="unknown",
            backend_name=None,
            using_fake_backend=bool(self.config.use_fake_backend),
            details={},
        )
        if self.config.noise_mode == "backend_scheduled":
            backend_obj, backend_name, using_fake = _resolve_noise_backend(self.config)
            self._backend_target = backend_obj
            self.backend_info = NoiseBackendInfo(
                noise_mode=str(self.config.noise_mode),
                estimator_kind="fake_backend.run(counts)",
                backend_name=str(backend_name),
                using_fake_backend=bool(using_fake),
                details={
                    "shots": int(self.config.shots),
                    "compiled_mode": "backend_scheduled",
                    "transpile_optimization_level": 1,
                    "transpile_seed": int(self.config.seed),
                    "mitigation": dict(mitigation_cfg),
                    "symmetry_mitigation": dict(symmetry_cfg),
                    "local_dynamical_decoupling": {
                        "requested": bool(mitigation_cfg.get("dd_sequence")),
                        "applied": False,
                        "sequence": mitigation_cfg.get("dd_sequence", None),
                        "reason": "not_evaluated",
                        "scheduling_method": "alap",
                        "timing_supported": False,
                    },
                    "aer_failed": False,
                    "fallback_used": False,
                },
            )
        else:
            try:
                self._estimator, self._session, self.backend_info, self._backend_target = _build_estimator(self.config)
            except Exception as exc:
                if self._can_fallback_from_error(exc):
                    self._activate_sampler_fallback(reason=str(exc), aer_failed=True)
                else:
                    raise
        self._update_backend_details(execution_surface="expectation_v1")
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
        self._closed = True

    def __enter__(self) -> "ExpectationOracle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _update_backend_details(self, **updates: Any) -> None:
        details = dict(getattr(self.backend_info, "details", {}))
        details.update(updates)
        self.backend_info = NoiseBackendInfo(
            noise_mode=str(self.backend_info.noise_mode),
            estimator_kind=str(self.backend_info.estimator_kind),
            backend_name=self.backend_info.backend_name,
            using_fake_backend=bool(self.backend_info.using_fake_backend),
            details=details,
        )

    def _snapshot_backend_info(self, **detail_updates: Any) -> dict[str, Any]:
        details = dict(getattr(self.backend_info, "details", {}))
        details.update(detail_updates)
        return {
            "noise_mode": str(self.backend_info.noise_mode),
            "estimator_kind": str(self.backend_info.estimator_kind),
            "backend_name": self.backend_info.backend_name,
            "using_fake_backend": bool(self.backend_info.using_fake_backend),
            "details": details,
        }

    def _set_symmetry_mitigation_details(self, details_map: Mapping[str, Any]) -> None:
        self._update_backend_details(symmetry_mitigation=dict(details_map))

    def _maybe_evaluate_symmetry_mitigated(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
    ) -> OracleEstimate | None:
        symmetry_cfg = normalize_symmetry_mitigation_config(
            getattr(self.config, "symmetry_mitigation", "off")
        )
        requested_mode = str(symmetry_cfg.get("mode", "off"))
        details: dict[str, Any] = {
            "requested_mode": str(requested_mode),
            "applied_mode": str(requested_mode),
            "executed": False,
            "eligible": False,
            "fallback_reason": "",
            "retained_fraction_mean": None,
            "retained_fraction_samples": [],
            "sector_probability_mean": None,
            "sector_probability_samples": [],
            "sector_values": {
                "sector_n_up": symmetry_cfg.get("sector_n_up", None),
                "sector_n_dn": symmetry_cfg.get("sector_n_dn", None),
            },
            "estimator_form": "none",
        }
        if requested_mode in {"off", "verify_only"}:
            self._set_symmetry_mitigation_details(details)
            return None
        if str(self.config.noise_mode) == "backend_scheduled":
            details["applied_mode"] = "verify_only"
            details["fallback_reason"] = "backend_scheduled_symmetry_not_supported"
            self._set_symmetry_mitigation_details(details)
            return None
        if not _observable_is_diagonal(observable):
            details["applied_mode"] = "verify_only"
            details["fallback_reason"] = "observable_not_diagonal"
            self._set_symmetry_mitigation_details(details)
            return None
        if str(self.config.noise_mode) == "runtime":
            details["applied_mode"] = "verify_only"
            details["fallback_reason"] = "runtime_counts_path_unavailable"
            self._set_symmetry_mitigation_details(details)
            return None
        required_keys = ("num_sites", "sector_n_up", "sector_n_dn")
        if any(symmetry_cfg.get(key, None) is None for key in required_keys):
            details["applied_mode"] = "verify_only"
            details["fallback_reason"] = "incomplete_sector_config"
            self._set_symmetry_mitigation_details(details)
            return None
        vals: list[float] = []
        retained: list[float] = []
        repeats = max(1, int(self.config.oracle_repeats))
        try:
            for rep in range(repeats):
                if str(self.config.noise_mode) == "ideal":
                    if requested_mode == "projector_renorm_v1":
                        val, retained_fraction = _exact_projector_renorm_diagonal_expectation(
                            circuit,
                            observable,
                            symmetry_cfg,
                        )
                    else:
                        val, retained_fraction = _exact_postselected_diagonal_expectation(
                            circuit,
                            observable,
                            symmetry_cfg,
                        )
                else:
                    counts = _sample_measurement_counts(circuit, self.config, repeat_idx=int(rep))
                    if requested_mode == "projector_renorm_v1":
                        val, retained_fraction = _projector_renorm_diagonal_expectation_from_counts(
                            counts,
                            observable,
                            n_qubits=int(circuit.num_qubits),
                            symmetry_cfg=symmetry_cfg,
                        )
                    else:
                        kept_counts, retained_fraction = _postselected_counts_and_fraction(
                            counts,
                            n_qubits=int(circuit.num_qubits),
                            symmetry_cfg=symmetry_cfg,
                        )
                        val = _diagonal_expectation_from_counts(kept_counts, observable)
                vals.append(float(val))
                retained.append(float(retained_fraction))
        except Exception as exc:
            if self._can_fallback_from_error(exc):
                self._activate_sampler_fallback(reason=str(exc), aer_failed=True)
            details["applied_mode"] = "verify_only"
            details["fallback_reason"] = str(exc)
            self._set_symmetry_mitigation_details(details)
            return None
        arr = np.asarray(vals, dtype=float)
        stdev = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        stderr = float(stdev / np.sqrt(float(arr.size))) if arr.size > 0 else float("nan")
        agg = float(np.median(arr)) if self.config.oracle_aggregate == "median" else float(np.mean(arr))
        details.update(
            {
                "applied_mode": str(requested_mode),
                "executed": True,
                "eligible": True,
                "fallback_reason": "",
                "retained_fraction_mean": (float(np.mean(retained)) if retained else None),
                "retained_fraction_samples": [float(x) for x in retained],
                "sector_probability_mean": (float(np.mean(retained)) if retained else None),
                "sector_probability_samples": [float(x) for x in retained],
                "estimator_form": (
                    "postselected_bitstring_average"
                    if requested_mode == "postselect_diag_v1"
                    else "projector_ratio_diag_v1"
                ),
            }
        )
        self._set_symmetry_mitigation_details(details)
        return OracleEstimate(
            mean=agg,
            std=stdev,
            stdev=stdev,
            stderr=stderr,
            n_samples=int(arr.size),
            raw_values=[float(x) for x in arr.tolist()],
            aggregate=self.config.oracle_aggregate,
        )

    def _get_backend_scheduled_base(self, circuit: QuantumCircuit) -> dict[str, Any]:
        cache_key = int(id(circuit))
        cached = self._compiled_base_cache.get(cache_key, None)
        if cached is not None:
            if cache_key not in self._compiled_base_cache_hit_logged:
                self._compiled_base_cache_hit_logged.add(cache_key)
                _oracle_compile_heartbeat(
                    "noise_oracle_backend_scheduled_compile_cache_hit",
                    noise_mode=str(self.config.noise_mode),
                    backend_name=self.backend_info.backend_name,
                    transpile_seed=int(
                        self.config.seed if self.config.seed_transpiler is None else self.config.seed_transpiler
                    ),
                    optimization_level=int(self.config.transpile_optimization_level),
                    cache_hit=True,
                    circuit=circuit,
                    compiled_payload=cached,
                )
            return cached
        if self._backend_target is None:
            raise RuntimeError("backend_scheduled backend target is unavailable.")
        transpile_seed = int(self.config.seed if self.config.seed_transpiler is None else self.config.seed_transpiler)
        optimization_level = int(self.config.transpile_optimization_level)
        compile_t0 = time.perf_counter()
        _oracle_compile_heartbeat(
            "noise_oracle_backend_scheduled_compile_start",
            noise_mode=str(self.config.noise_mode),
            backend_name=self.backend_info.backend_name,
            transpile_seed=int(transpile_seed),
            optimization_level=int(optimization_level),
            cache_hit=False,
            circuit=circuit,
        )
        cached = compile_circuit_for_local_backend(
            circuit,
            self._backend_target,
            seed_transpiler=transpile_seed,
            optimization_level=optimization_level,
        )
        _oracle_compile_heartbeat(
            "noise_oracle_backend_scheduled_compile_done",
            noise_mode=str(self.config.noise_mode),
            backend_name=self.backend_info.backend_name,
            transpile_seed=int(transpile_seed),
            optimization_level=int(optimization_level),
            cache_hit=False,
            circuit=circuit,
            elapsed_s=float(time.perf_counter() - compile_t0),
            compiled_payload=cached,
        )
        self._compiled_base_cache[cache_key] = cached
        compiled = cached["compiled"]
        self._update_backend_details(
            transpile_optimization_level=int(optimization_level),
            transpile_seed=int(transpile_seed),
            layout_physical_qubits=[int(x) for x in cached["logical_to_physical"]],
            compiled_num_qubits=int(cached["compiled_num_qubits"]),
            **_compiled_metrics_payload(compiled),
        )
        return cached

    def _get_backend_scheduled_repeat_base(
        self,
        compiled_base: QuantumCircuit,
        *,
        repeat_idx: int,
    ) -> tuple[QuantumCircuit, dict[str, Any]]:
        mitigation_cfg = normalize_mitigation_config(getattr(self.config, "mitigation", "none"))
        if not _local_gate_twirling_enabled(mitigation_cfg):
            return compiled_base, {
                "requested": False,
                "applied": False,
                "reason": "local_gate_twirling_disabled",
                "engine": "none",
                "seed": None,
            }
        cache_key = (int(id(compiled_base)), int(repeat_idx))
        cached = self._backend_scheduled_repeat_base_cache.get(cache_key, None)
        if cached is not None:
            return cached["compiled"], dict(cached["details"])
        twirl_seed = int(self.config.seed) + int(repeat_idx)
        twirled_base, details = _twirl_compiled_two_qubit_base(
            compiled_base,
            backend_target=self._backend_target,
            seed=int(twirl_seed),
        )
        self._backend_scheduled_repeat_base_cache[cache_key] = {
            "compiled": twirled_base,
            "details": dict(details),
        }
        return twirled_base, dict(details)

    def _get_runtime_isa_base(self, circuit: QuantumCircuit) -> dict[str, Any]:
        cache_key = int(id(circuit))
        cached = self._compiled_base_cache.get(cache_key, None)
        if cached is not None:
            if cache_key not in self._compiled_base_cache_hit_logged:
                self._compiled_base_cache_hit_logged.add(cache_key)
                _oracle_compile_heartbeat(
                    "noise_oracle_runtime_compile_cache_hit",
                    noise_mode=str(self.config.noise_mode),
                    backend_name=self.backend_info.backend_name,
                    transpile_seed=int(
                        self.config.seed if self.config.seed_transpiler is None else self.config.seed_transpiler
                    ),
                    optimization_level=int(self.config.transpile_optimization_level),
                    cache_hit=True,
                    circuit=circuit,
                    compiled_payload=cached,
                )
            return cached
        if self._backend_target is None:
            raise RuntimeError("runtime backend target is unavailable.")
        transpile_seed = int(self.config.seed if self.config.seed_transpiler is None else self.config.seed_transpiler)
        optimization_level = int(self.config.transpile_optimization_level)
        compile_t0 = time.perf_counter()
        _oracle_compile_heartbeat(
            "noise_oracle_runtime_compile_start",
            noise_mode=str(self.config.noise_mode),
            backend_name=self.backend_info.backend_name,
            transpile_seed=int(transpile_seed),
            optimization_level=int(optimization_level),
            cache_hit=False,
            circuit=circuit,
        )
        cached = compile_circuit_for_local_backend(
            circuit,
            self._backend_target,
            seed_transpiler=transpile_seed,
            optimization_level=optimization_level,
        )
        _oracle_compile_heartbeat(
            "noise_oracle_runtime_compile_done",
            noise_mode=str(self.config.noise_mode),
            backend_name=self.backend_info.backend_name,
            transpile_seed=int(transpile_seed),
            optimization_level=int(optimization_level),
            cache_hit=False,
            circuit=circuit,
            elapsed_s=float(time.perf_counter() - compile_t0),
            compiled_payload=cached,
        )
        self._compiled_base_cache[cache_key] = cached
        compiled = cached["compiled"]
        self._update_backend_details(
            transpile_optimization_level=int(optimization_level),
            transpile_seed=int(transpile_seed),
            layout_physical_qubits=[int(x) for x in cached["logical_to_physical"]],
            compiled_num_qubits=int(cached["compiled_num_qubits"]),
            **_compiled_metrics_payload(compiled),
        )
        return cached

    def _runtime_group_sampling_supported(self) -> bool:
        mitigation_cfg = normalize_mitigation_config(getattr(self.config, "mitigation", "none"))
        symmetry_cfg = normalize_symmetry_mitigation_config(
            getattr(self.config, "symmetry_mitigation", "off")
        )
        return bool(
            str(self.config.noise_mode) == "runtime"
            and str(mitigation_cfg.get("mode", "none")) == "none"
            and str(symmetry_cfg.get("mode", "off")) in {"off", "verify_only"}
        )

    def _get_runtime_sampler(self) -> Any:
        if self._runtime_sampler is not None:
            return self._runtime_sampler
        if str(self.config.noise_mode) != "runtime":
            raise ValueError("_get_runtime_sampler requires runtime noise mode.")
        if not self._runtime_group_sampling_supported():
            raise ValueError(
                "runtime grouped sampling currently requires mitigation='none' and symmetry_mitigation in {'off','verify_only'}."
            )
        try:
            from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2
        except Exception as exc:
            raise RuntimeError(
                "runtime grouped sampling requires qiskit-ibm-runtime SamplerV2."
            ) from exc
        runtime_mode = self._session if self._session is not None else self._backend_target
        sampler = RuntimeSamplerV2(mode=runtime_mode)
        runtime_profile_cfg = normalize_runtime_estimator_profile_config(
            getattr(self.config, "runtime_profile", "legacy_runtime_v0")
        )
        self._runtime_sampler_configured_details = _configure_runtime_sampler_options(
            sampler,
            cfg=self.config,
            runtime_profile_cfg=runtime_profile_cfg,
        )
        self._runtime_sampler = sampler
        return sampler

    def _get_runtime_measured_isa(
        self,
        circuit: QuantumCircuit,
        measurement_basis_ixyz: str,
    ) -> dict[str, Any]:
        cache_key = (int(id(circuit)), str(measurement_basis_ixyz).upper())
        cached = self._runtime_measured_isa_cache.get(cache_key, None)
        if cached is not None:
            return cached
        if self._backend_target is None:
            raise RuntimeError("runtime backend target is unavailable.")
        measured, active_logical = _sparse_term_measurement_circuit(
            circuit,
            str(measurement_basis_ixyz).upper(),
        )
        transpile_seed = int(
            self.config.seed
            if self.config.seed_transpiler is None
            else self.config.seed_transpiler
        )
        optimization_level = int(self.config.transpile_optimization_level)
        compiled = compile_circuit_for_local_backend(
            measured,
            self._backend_target,
            seed_transpiler=transpile_seed,
            optimization_level=optimization_level,
        )
        cached = {
            "compiled": compiled["compiled"],
            "active_logical_qubits": tuple(int(q) for q in active_logical),
            "basis_label": str(measurement_basis_ixyz).upper(),
        }
        self._runtime_measured_isa_cache[cache_key] = cached
        return cached

    def _ensure_mthree_calibration(self, active_physical_qubits: Sequence[int]) -> None:
        self._mthree_mitigator, _cache_miss = _ensure_mthree_calibration_cached(
            backend_target=self._backend_target,
            mitigator=self._mthree_mitigator,
            calibrated_qubits=self._mthree_calibrated_qubits,
            active_physical_qubits=active_physical_qubits,
            shots=int(self.config.shots),
            mthree_module=self._mthree_module,
        )

    def _get_backend_scheduled_full_noise_model(self) -> Any:
        if self._backend_scheduled_noise_model is None:
            if self._backend_target is None:
                raise RuntimeError("backend_scheduled backend target is unavailable.")
            from qiskit_aer.noise import NoiseModel

            self._backend_scheduled_noise_model = NoiseModel.from_backend(self._backend_target)
        return self._backend_scheduled_noise_model

    def _get_backend_scheduled_local_dd_retry_target(self) -> tuple[Any, dict[str, Any]]:
        if self._backend_scheduled_local_dd_retry_target is not None:
            return self._backend_scheduled_local_dd_retry_target, dict(
                self._backend_scheduled_local_dd_retry_details or {}
            )
        if self._backend_target is None:
            raise LocalDDSupportError("backend_scheduled backend target is unavailable.")
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel

        properties, sanitization = _sanitize_backend_relaxation_properties_for_local_dd(
            self._backend_target
        )
        wrapped_backend = _BackendPropertiesOverrideWrapper(self._backend_target, properties)
        try:
            noise_model = NoiseModel.from_backend(wrapped_backend)
        except Exception as exc:
            raise LocalDDSupportError(
                f"Failed to build sanitized local DD retry noise model: {type(exc).__name__}: {exc}"
            ) from exc
        target = AerSimulator(
            noise_model=noise_model,
            seed_simulator=int(self.config.seed),
        )
        details = {
            "execution_target_kind": "aer_sanitized_from_backend",
            "noise_model_basis_gates": list(getattr(noise_model, "basis_gates", []) or []),
            "noise_model_sanitization": dict(sanitization),
        }
        self._backend_scheduled_local_dd_retry_target = target
        self._backend_scheduled_local_dd_retry_details = dict(details)
        return target, dict(details)

    def _get_backend_scheduled_attribution_target(
        self,
        slice_name: str,
    ) -> tuple[Any, dict[str, Any]]:
        key = str(slice_name).strip().lower()
        if key not in BACKEND_SCHEDULED_ATTRIBUTION_SLICES:
            raise ValueError(
                f"Unsupported backend_scheduled attribution slice {slice_name!r}; "
                f"expected one of {list(BACKEND_SCHEDULED_ATTRIBUTION_SLICES)}."
            )
        cached = self._backend_scheduled_attribution_targets.get(key, None)
        if cached is not None:
            return cached["target"], dict(cached["details"])
        if self._backend_target is None:
            raise RuntimeError("backend_scheduled backend target is unavailable.")

        if key == "full":
            target = self._backend_target
            details = {
                "attribution_slice": "full",
                "shared_compile_reused": True,
                "execution_target_kind": "fake_backend.run",
                "components": {
                    "gate_stateprep": True,
                    "readout": True,
                },
            }
        else:
            from qiskit_aer import AerSimulator

            full_model = self._get_backend_scheduled_full_noise_model()
            if key == "readout_only":
                model = _copy_noise_model_components(
                    full_model,
                    include_quantum=False,
                    include_readout=True,
                )
                components = {"gate_stateprep": False, "readout": True}
            else:
                model = _copy_noise_model_components(
                    full_model,
                    include_quantum=True,
                    include_readout=False,
                )
                components = {"gate_stateprep": True, "readout": False}
            target = AerSimulator(
                noise_model=model,
                seed_simulator=int(self.config.seed),
            )
            details = {
                "attribution_slice": str(key),
                "shared_compile_reused": True,
                "execution_target_kind": "AerSimulator",
                "noise_model_basis_gates": list(getattr(model, "basis_gates", []) or []),
                "components": dict(components),
            }
        self._backend_scheduled_attribution_targets[key] = {
            "target": target,
            "details": dict(details),
        }
        return target, dict(details)

    def _build_backend_scheduled_measured_term_circuit(
        self,
        compiled_base: QuantumCircuit,
        logical_to_physical: Sequence[int],
        pauli_label_ixyz: str,
    ) -> tuple[QuantumCircuit, tuple[int, ...], dict[str, Any]]:
        label = str(pauli_label_ixyz).upper()
        active_logical = _active_logical_qubits_for_label(label)
        active_physical = tuple(int(logical_to_physical[q]) for q in active_logical)
        qc = compiled_base.copy()
        if active_physical:
            creg = ClassicalRegister(len(active_physical), "m")
            qc.add_register(creg)
            for q in active_logical:
                op = label[int(len(label)) - 1 - int(q)]
                phys = int(logical_to_physical[int(q)])
                if op == "X":
                    qc.h(phys)
                elif op == "Y":
                    qc.sdg(phys)
                    qc.h(phys)
                elif op in {"I", "Z"}:
                    pass
                else:
                    raise ValueError(f"Unsupported Pauli op '{op}' in '{label}'")
            for idx, phys in enumerate(active_physical):
                qc.measure(int(phys), creg[int(idx)])
        details = {
            "active_logical_qubits": [int(x) for x in active_logical],
            "active_physical_qubits": [int(x) for x in active_physical],
            "pauli_weight": int(_active_pauli_weight(label)),
        }
        return qc, active_physical, details

    def _maybe_apply_local_dd_to_measured_term_circuit(
        self,
        circuit: QuantumCircuit,
        *,
        logical_to_physical: Sequence[int],
        dd_sequence: str | None,
    ) -> tuple[QuantumCircuit, dict[str, Any]]:
        sequence = None if dd_sequence in {None, "", "none"} else str(dd_sequence).strip().upper()
        base_details = {
            "requested": bool(sequence),
            "applied": False,
            "sequence": sequence,
            "reason": "disabled" if sequence is None else "not_applied",
            "scheduling_method": "alap",
            "timing_supported": False,
        }
        if sequence is None:
            return circuit, base_details
        if sequence != "XPXM":
            raise LocalDDSupportError(
                f"Unsupported local backend_scheduled DD sequence {sequence!r}; expected 'XpXm'."
            )
        if self._backend_target is None:
            raise LocalDDSupportError("backend_scheduled backend target is unavailable.")
        target = getattr(self._backend_target, "target", None)
        if target is None:
            raise LocalDDSupportError("backend target timing metadata is unavailable for local DD.")
        try:
            from qiskit.circuit.library import XGate
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
        except Exception as exc:
            raise LocalDDSupportError(
                "Qiskit scheduling/DD passes are unavailable for local backend_scheduled DD."
            ) from exc

        dd_qubits = [int(q) for q in tuple(logical_to_physical)]
        circuit_for_dd = _rewrite_local_dd_measurement_basis_ops(circuit)
        base_counts = dict(circuit_for_dd.count_ops())
        try:
            pm = PassManager(
                [
                    ALAPScheduleAnalysis(target=target),
                    PadDynamicalDecoupling(
                        target=target,
                        dd_sequence=[XGate(), XGate()],
                        qubits=list(dd_qubits),
                    ),
                ]
            )
            dd_circuit = pm.run(circuit_for_dd)
        except Exception as exc:
            raise LocalDDSupportError(
                f"Failed to schedule/apply local DD on measured term circuit: {type(exc).__name__}: {exc}"
            ) from exc

        dd_counts = dict(dd_circuit.count_ops())
        added_x = int(dd_counts.get("x", 0)) - int(base_counts.get("x", 0))
        applied = bool(added_x > 0)
        details = {
            "requested": True,
            "applied": bool(applied),
            "sequence": str(sequence),
            "reason": ("" if applied else "no_idle_windows_or_insertions"),
            "scheduling_method": "alap",
            "timing_supported": True,
            "dd_qubits": [int(q) for q in dd_qubits],
            "added_x": int(max(0, added_x)),
            "added_y": 0,
        }
        return dd_circuit, details

    def _backend_scheduled_term_counts(
        self,
        compiled_base: QuantumCircuit,
        logical_to_physical: Sequence[int],
        pauli_label_ixyz: str,
        *,
        repeat_idx: int,
        execution_target: Any | None = None,
    ) -> tuple[dict[str, int], tuple[int, ...], dict[str, Any]]:
        target = self._backend_target if execution_target is None else execution_target
        if target is None:
            raise RuntimeError("backend_scheduled backend target is unavailable.")
        mitigation_cfg = normalize_mitigation_config(getattr(self.config, "mitigation", "none"))
        qc, active_physical, details = self._build_backend_scheduled_measured_term_circuit(
            compiled_base,
            logical_to_physical,
            pauli_label_ixyz,
        )
        qc, dd_details = self._maybe_apply_local_dd_to_measured_term_circuit(
            qc,
            logical_to_physical=logical_to_physical,
            dd_sequence=mitigation_cfg.get("dd_sequence", None),
        )
        dd_requested = bool(dd_details.get("requested", False))
        seed_simulator = int(self.config.seed) + int(repeat_idx)
        dd_retry_details: dict[str, Any] | None = None
        if dd_requested and execution_target is None and self._backend_scheduled_local_dd_retry_engaged:
            target, dd_retry_details = self._get_backend_scheduled_local_dd_retry_target()
        try:
            result = target.run(
                qc,
                shots=int(self.config.shots),
                seed_simulator=int(seed_simulator),
            ).result()
        except Exception as exc:
            if not (dd_requested and execution_target is None and _is_invalid_t2_relaxation_error(exc)):
                raise
            try:
                target, dd_retry_details = self._get_backend_scheduled_local_dd_retry_target()
                self._backend_scheduled_local_dd_retry_engaged = True
                result = target.run(
                    qc,
                    shots=int(self.config.shots),
                    seed_simulator=int(seed_simulator),
                ).result()
            except Exception as retry_exc:
                raise LocalDDSupportError(
                    "Failed to execute backend_scheduled local DD with sanitized retry target: "
                    f"{type(retry_exc).__name__}: {retry_exc}"
                ) from retry_exc
        counts = result.get_counts()
        if isinstance(counts, list):
            counts = counts[0]
        if dd_retry_details is not None:
            dd_details = {**dict(dd_details), **dict(dd_retry_details)}
        details.update(
            {
                "compiled_depth": int(qc.depth() or 0),
                "compiled_size": int(qc.size()),
                "local_dynamical_decoupling": dict(dd_details),
            }
        )
        return dict(counts), active_physical, details

    def _evaluate_backend_scheduled_with_target(
        self,
        compiled_base: QuantumCircuit,
        logical_to_physical: Sequence[int],
        observable: SparsePauliOp,
        *,
        execution_target: Any | None = None,
        attribution_slice: str | None = None,
        target_details: Mapping[str, Any] | None = None,
    ) -> tuple[OracleEstimate, dict[str, Any]]:
        mitigation_cfg = normalize_mitigation_config(getattr(self.config, "mitigation", "none"))
        mitigation_mode = str(mitigation_cfg.get("mode", "none"))
        eval_t0 = time.perf_counter()
        timing_repeat_base_s = 0.0
        timing_term_counts_s = 0.0
        timing_readout_cal_s = 0.0
        timing_readout_apply_s = 0.0
        readout_calibration_cache_misses = 0
        observable_terms_evaluated = 0
        vals: list[float] = []
        last_twirling_details: dict[str, Any] = {
            "requested": bool(_local_gate_twirling_enabled(mitigation_cfg)),
            "applied": False,
            "reason": "not_evaluated",
            "engine": "none",
            "seed": None,
        }
        last_readout_details: dict[str, Any] = {
            "mode": str(mitigation_mode),
            "strategy": mitigation_cfg.get("local_readout_strategy", None),
            "applied": False,
        }
        last_dd_details: dict[str, Any] = {
            "requested": bool(mitigation_cfg.get("dd_sequence")),
            "applied": False,
            "sequence": mitigation_cfg.get("dd_sequence", None),
            "reason": "not_evaluated",
            "scheduling_method": "alap",
            "timing_supported": False,
        }
        for rep in range(max(1, int(self.config.oracle_repeats))):
            repeat_base_t0 = time.perf_counter()
            repeat_base, twirling_details = self._get_backend_scheduled_repeat_base(
                compiled_base,
                repeat_idx=int(rep),
            )
            timing_repeat_base_s += float(time.perf_counter() - repeat_base_t0)
            last_twirling_details = dict(twirling_details)
            total = 0.0 + 0.0j
            for label, coeff in observable.to_list():
                coeff_c = complex(coeff)
                label_s = str(label).upper()
                if all(ch == "I" for ch in label_s):
                    total += coeff_c
                    continue
                observable_terms_evaluated += 1
                term_counts_t0 = time.perf_counter()
                counts, active_physical, term_details = self._backend_scheduled_term_counts(
                    repeat_base,
                    logical_to_physical,
                    label_s,
                    repeat_idx=int(rep),
                    execution_target=execution_target,
                )
                timing_term_counts_s += float(time.perf_counter() - term_counts_t0)
                last_dd_details = dict(term_details.get("local_dynamical_decoupling", {}))
                if mitigation_mode == "readout":
                    calibration_miss = tuple(int(x) for x in active_physical) not in self._mthree_calibrated_qubits
                    readout_cal_t0 = time.perf_counter()
                    self._ensure_mthree_calibration(active_physical)
                    timing_readout_cal_s += float(time.perf_counter() - readout_cal_t0)
                    if calibration_miss:
                        readout_calibration_cache_misses += 1
                    readout_apply_t0 = time.perf_counter()
                    quasi, mitigation_details = _apply_mthree_readout_correction(
                        mitigator=self._mthree_mitigator,
                        counts=counts,
                        active_physical_qubits=active_physical,
                    )
                    timing_readout_apply_s += float(time.perf_counter() - readout_apply_t0)
                    exp_lbl = _parity_expectation_from_quasi(quasi, len(active_physical))
                    last_readout_details = {
                        "mode": "readout",
                        "strategy": str(mitigation_cfg.get("local_readout_strategy", "mthree")),
                        "applied": True,
                        "active_physical_qubits": [int(x) for x in active_physical],
                        "term_details": dict(term_details),
                        "solver_method": str(mitigation_details.get("method", "")),
                        "solver_time_s": float(mitigation_details.get("time", float("nan"))),
                        "dimension": int(mitigation_details.get("dimension", 0)),
                        "mitigation_overhead": float(
                            mitigation_details.get("mitigation_overhead", float("nan"))
                        ),
                        "negative_mass": float(mitigation_details.get("negative_mass", 0.0)),
                        "shots": int(mitigation_details.get("shots", int(self.config.shots))),
                        "calibration_cache_size": int(len(self._mthree_calibrated_qubits)),
                    }
                else:
                    exp_lbl = _parity_expectation_from_active_counts(counts, len(active_physical))
                    last_readout_details = {
                        "mode": mitigation_mode,
                        "strategy": mitigation_cfg.get("local_readout_strategy", None),
                        "applied": False,
                        "active_physical_qubits": [int(x) for x in active_physical],
                        "term_details": dict(term_details),
                    }
                total += coeff_c * complex(float(exp_lbl), 0.0)
            vals.append(float(np.real(total)))
        arr = np.asarray(vals, dtype=float)
        stdev = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        stderr = float(stdev / np.sqrt(float(arr.size))) if arr.size > 0 else float("nan")
        agg = float(np.median(arr)) if self.config.oracle_aggregate == "median" else float(np.mean(arr))
        details = {
            "readout_mitigation": dict(last_readout_details),
            "local_gate_twirling": dict(last_twirling_details),
            "local_dynamical_decoupling": dict(last_dd_details),
            "debug_timing": {
                "elapsed_total_s": float(time.perf_counter() - eval_t0),
                "repeat_base_s": float(timing_repeat_base_s),
                "term_counts_s": float(timing_term_counts_s),
                "readout_calibration_s": float(timing_readout_cal_s),
                "readout_apply_s": float(timing_readout_apply_s),
                "readout_calibration_cache_misses": int(readout_calibration_cache_misses),
                "observable_terms_evaluated": int(observable_terms_evaluated),
                "oracle_repeats": int(max(1, int(self.config.oracle_repeats))),
                "mitigation_mode": str(mitigation_mode),
            },
        }
        _oracle_debug_log(
            "noise_oracle_backend_scheduled_eval_timing",
            elapsed_total_s=float(details["debug_timing"]["elapsed_total_s"]),
            repeat_base_s=float(timing_repeat_base_s),
            term_counts_s=float(timing_term_counts_s),
            readout_calibration_s=float(timing_readout_cal_s),
            readout_apply_s=float(timing_readout_apply_s),
            readout_calibration_cache_misses=int(readout_calibration_cache_misses),
            observable_terms_evaluated=int(observable_terms_evaluated),
            oracle_repeats=int(max(1, int(self.config.oracle_repeats))),
            mitigation_mode=str(mitigation_mode),
            attribution_slice=(None if attribution_slice is None else str(attribution_slice)),
        )
        if attribution_slice is not None:
            details["attribution_slice"] = str(attribution_slice)
            details["shared_compile_reused"] = True
        if target_details is not None:
            details.update(dict(target_details))
        return (
            OracleEstimate(
                mean=agg,
                std=stdev,
                stdev=stdev,
                stderr=stderr,
                n_samples=int(arr.size),
                raw_values=[float(x) for x in arr.tolist()],
                aggregate=self.config.oracle_aggregate,
            ),
            details,
        )

    def collect_backend_scheduled_term_sample(
        self,
        circuit: QuantumCircuit,
        pauli_label_ixyz: str,
        *,
        repeat_idx: int = 0,
    ) -> dict[str, Any]:
        group_payload = self.collect_backend_scheduled_group_sample(
            circuit,
            pauli_label_ixyz,
            repeat_idx=int(repeat_idx),
        )
        measured_logical = tuple(
            int(q) for q in group_payload.get("measured_logical_qubits", ())
        )
        expectation = (
            _parity_expectation_from_quasi(
                group_payload["quasi_probs"],
                len(measured_logical),
            )
            if group_payload.get("quasi_probs", None) is not None
            else _parity_expectation_from_active_counts(
                group_payload["counts"],
                len(measured_logical),
            )
        )
        return {
            **dict(group_payload),
            "expectation": float(expectation),
        }

    def collect_group_sample(
        self,
        circuit: QuantumCircuit,
        measurement_basis_ixyz: str,
        *,
        repeat_idx: int = 0,
    ) -> dict[str, Any]:
        mode = str(self.config.noise_mode)
        if mode == "backend_scheduled":
            return self.collect_backend_scheduled_group_sample(
                circuit,
                measurement_basis_ixyz,
                repeat_idx=int(repeat_idx),
            )
        if mode == "runtime":
            return self.collect_runtime_group_sample(
                circuit,
                measurement_basis_ixyz,
                repeat_idx=int(repeat_idx),
            )
        raise ValueError(
            f"collect_group_sample unsupported for noise mode {self.config.noise_mode!r}."
        )

    def collect_backend_scheduled_group_sample(
        self,
        circuit: QuantumCircuit,
        measurement_basis_ixyz: str,
        *,
        repeat_idx: int = 0,
    ) -> dict[str, Any]:
        if str(self.config.noise_mode) != "backend_scheduled":
            raise ValueError("collect_backend_scheduled_group_sample requires backend_scheduled noise mode.")
        mitigation_cfg = normalize_mitigation_config(getattr(self.config, "mitigation", "none"))
        mitigation_mode = str(mitigation_cfg.get("mode", "none"))
        base = self._get_backend_scheduled_base(circuit)
        repeat_base, twirling_details = self._get_backend_scheduled_repeat_base(
            base["compiled"],
            repeat_idx=int(repeat_idx),
        )
        counts, active_physical, term_details = self._backend_scheduled_term_counts(
            repeat_base,
            base["logical_to_physical"],
            measurement_basis_ixyz,
            repeat_idx=int(repeat_idx),
        )
        quasi_probs: dict[str, float] | None = None
        if mitigation_mode == "readout":
            self._ensure_mthree_calibration(active_physical)
            quasi, mitigation_details = _apply_mthree_readout_correction(
                mitigator=self._mthree_mitigator,
                counts=counts,
                active_physical_qubits=active_physical,
            )
            quasi_probs = {str(key): float(val) for key, val in dict(quasi).items()}
            readout_details = {
                "mode": "readout",
                "strategy": str(mitigation_cfg.get("local_readout_strategy", "mthree")),
                "applied": True,
                "active_physical_qubits": [int(x) for x in active_physical],
                "term_details": dict(term_details),
                "solver_method": str(mitigation_details.get("method", "")),
                "solver_time_s": float(mitigation_details.get("time", float("nan"))),
                "dimension": int(mitigation_details.get("dimension", 0)),
                "mitigation_overhead": float(mitigation_details.get("mitigation_overhead", float("nan"))),
                "negative_mass": float(mitigation_details.get("negative_mass", 0.0)),
                "shots": int(mitigation_details.get("shots", int(self.config.shots))),
                "calibration_cache_size": int(len(self._mthree_calibrated_qubits)),
            }
        else:
            readout_details = {
                "mode": mitigation_mode,
                "strategy": mitigation_cfg.get("local_readout_strategy", None),
                "applied": False,
                "active_physical_qubits": [int(x) for x in active_physical],
                "term_details": dict(term_details),
            }
        dd_details = dict(term_details.get("local_dynamical_decoupling", {}))
        self._update_backend_details(
            readout_mitigation=dict(readout_details),
            local_gate_twirling=dict(twirling_details),
            local_dynamical_decoupling=dict(dd_details),
        )
        return {
            "repeat_index": int(repeat_idx),
            "shots": int(self.config.shots),
            "counts": dict(counts),
            "basis_label": str(measurement_basis_ixyz).upper(),
            "measured_logical_qubits": list(term_details.get("active_logical_qubits", [])),
            "quasi_probs": (None if quasi_probs is None else dict(quasi_probs)),
            "term_details": dict(term_details),
            "readout_mitigation": dict(readout_details),
            "local_gate_twirling": dict(twirling_details),
            "local_dynamical_decoupling": dict(dd_details),
        }

    def collect_runtime_group_sample(
        self,
        circuit: QuantumCircuit,
        measurement_basis_ixyz: str,
        *,
        repeat_idx: int = 0,
    ) -> dict[str, Any]:
        if str(self.config.noise_mode) != "runtime":
            raise ValueError("collect_runtime_group_sample requires runtime noise mode.")
        sampler = self._get_runtime_sampler()
        measured = self._get_runtime_measured_isa(circuit, str(measurement_basis_ixyz).upper())
        execution = _run_sampler_job(
            sampler,
            measured["compiled"],
            shots=int(self.config.shots),
            repeat_index=int(repeat_idx),
        )
        counts = dict(execution.counts)
        runtime_jobs = [asdict(rec) for rec in execution.job_records]
        sampler_details = (
            {} if self._runtime_sampler_configured_details is None else dict(self._runtime_sampler_configured_details)
        )
        self._update_backend_details(
            runtime_sampler=dict(sampler_details),
            runtime_group_sampling={
                "basis_label": str(measurement_basis_ixyz).upper(),
                "active_logical_qubits": [int(q) for q in measured["active_logical_qubits"]],
                "shots": int(self.config.shots),
            },
            runtime_job_ids=[
                str(rec.get("job_id"))
                for rec in runtime_jobs
                if isinstance(rec, Mapping) and rec.get("job_id", None) not in {None, ""}
            ],
            runtime_jobs=list(runtime_jobs),
            readout_mitigation={
                "mode": "none",
                "requested_strategy": None,
                "engine": "none",
                "applied": False,
            },
        )
        return {
            "repeat_index": int(repeat_idx),
            "shots": int(self.config.shots),
            "counts": counts,
            "basis_label": str(measurement_basis_ixyz).upper(),
            "measured_logical_qubits": [int(q) for q in measured["active_logical_qubits"]],
            "quasi_probs": None,
            "term_details": {
                "active_logical_qubits": [int(q) for q in measured["active_logical_qubits"]],
                "active_physical_qubits": None,
                "pauli_weight": int(_active_pauli_weight(str(measurement_basis_ixyz).upper())),
                "label": str(measurement_basis_ixyz).upper(),
                "runtime_used_call_path": str(execution.used_call_path),
            },
            "readout_mitigation": {
                "mode": "none",
                "strategy": None,
                "applied": False,
            },
            "local_gate_twirling": {
                "requested": bool(
                    sampler_details.get("gate_twirling", False)
                    or sampler_details.get("measure_twirling", False)
                ),
                "applied": bool(
                    sampler_details.get("gate_twirling", False)
                    or sampler_details.get("measure_twirling", False)
                ),
                "engine": "runtime_sampler_options",
                "strategy": sampler_details.get("twirling_strategy", None),
            },
            "local_dynamical_decoupling": {
                "requested": bool(sampler_details.get("dd_enable", False)),
                "applied": bool(sampler_details.get("dd_enable", False)),
                "sequence": sampler_details.get("dd_sequence", None),
                "scheduling_method": "runtime_sampler_options",
            },
        }

    def _evaluate_backend_scheduled(self, circuit: QuantumCircuit, observable: SparsePauliOp) -> OracleEstimate:
        base = self._get_backend_scheduled_base(circuit)
        estimate, details = self._evaluate_backend_scheduled_with_target(
            base["compiled"],
            base["logical_to_physical"],
            observable,
        )
        self._update_backend_details(**details)
        return estimate

    def _evaluate_runtime(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
        *,
        runtime_job_observer: Callable[[dict[str, Any]], None] | None = None,
        runtime_trace_context: Mapping[str, Any] | None = None,
    ) -> OracleEstimate:
        base = self._get_runtime_isa_base(circuit)
        isa_circuit = base["compiled"]
        isa_observable = _apply_observable_layout_for_compiled_circuit(observable, isa_circuit)

        mitigation_cfg = normalize_mitigation_config(getattr(self.config, "mitigation", "none"))
        runtime_readout_details = {
            "mode": str(mitigation_cfg.get("mode", "none")),
            "requested_strategy": mitigation_cfg.get("local_readout_strategy", None),
            "engine": self.backend_info.details.get("runtime_mitigation", {}).get("engine", "none"),
            "applied": bool(self.backend_info.details.get("runtime_mitigation", {}).get("applied", False)),
        }

        vals: list[float] = []
        runtime_jobs: list[dict[str, Any]] = []
        repeats = max(1, int(self.config.oracle_repeats))
        for rep in range(repeats):
            if self._sampler_fallback is not None:
                val = _run_sampler_fallback_job(self._sampler_fallback, circuit, observable)
                vals.append(float(np.real(val)))
                continue
            try:
                execution = _run_estimator_job(
                    self._estimator,
                    isa_circuit,
                    isa_observable,
                    repeat_index=int(rep),
                    job_observer=runtime_job_observer,
                    job_context=runtime_trace_context,
                )
                vals.append(float(np.real(execution.expectation_value)))
                runtime_jobs.extend(asdict(rec) for rec in execution.job_records)
            except SubmittedRuntimeJobError as exc:
                runtime_jobs.append(asdict(exc.record))
                raise
            except Exception as exc:
                if self._can_fallback_from_error(exc):
                    self._activate_sampler_fallback(reason=str(exc), aer_failed=True)
                    val = _run_sampler_fallback_job(self._sampler_fallback, circuit, observable)
                    vals.append(float(np.real(val)))
                else:
                    raise

        self._update_backend_details(
            isa_observable_num_qubits=int(isa_observable.num_qubits),
            readout_mitigation=dict(runtime_readout_details),
            runtime_job_ids=[
                str(rec.get("job_id"))
                for rec in runtime_jobs
                if isinstance(rec, Mapping) and rec.get("job_id", None) not in {None, ""}
            ],
            runtime_jobs=list(runtime_jobs),
            last_runtime_trace_context=(
                {} if runtime_trace_context is None else {str(k): v for k, v in dict(runtime_trace_context).items()}
            ),
        )
        return _oracle_estimate_from_samples(vals, aggregate=str(self.config.oracle_aggregate))

    def evaluate_backend_scheduled_attribution(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
        *,
        slices: Sequence[str] = BACKEND_SCHEDULED_ATTRIBUTION_SLICES,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("ExpectationOracle is closed.")
        if str(self.config.noise_mode) != "backend_scheduled":
            raise ValueError("backend_scheduled attribution requires noise_mode='backend_scheduled'.")
        if not bool(self.config.use_fake_backend):
            raise ValueError("backend_scheduled attribution requires use_fake_backend=True.")
        mitigation_cfg = normalize_mitigation_config(getattr(self.config, "mitigation", "none"))
        symmetry_cfg = normalize_symmetry_mitigation_config(
            getattr(self.config, "symmetry_mitigation", "off")
        )
        if str(mitigation_cfg.get("mode", "none")) != "none":
            raise ValueError("backend_scheduled attribution requires mitigation mode 'none'.")
        if bool(mitigation_cfg.get("local_gate_twirling", False)):
            raise ValueError("backend_scheduled attribution requires local gate twirling off.")
        if mitigation_cfg.get("dd_sequence", None) not in {None, "", "none"}:
            raise ValueError("backend_scheduled attribution requires local DD off.")
        if str(symmetry_cfg.get("mode", "off")) != "off":
            raise ValueError("backend_scheduled attribution requires symmetry mitigation 'off'.")

        requested_slices = tuple(str(x).strip().lower() for x in slices)
        for slice_name in requested_slices:
            if slice_name not in BACKEND_SCHEDULED_ATTRIBUTION_SLICES:
                raise ValueError(
                    f"Unsupported backend_scheduled attribution slice {slice_name!r}; "
                    f"expected one of {list(BACKEND_SCHEDULED_ATTRIBUTION_SLICES)}."
                )

        base = self._get_backend_scheduled_base(circuit)
        compiled_base = base["compiled"]
        logical_to_physical = base["logical_to_physical"]
        shared_compile = {
            "shared_transpile": True,
            "backend_name": self.backend_info.backend_name,
            "using_fake_backend": bool(self.backend_info.using_fake_backend),
            "transpile_optimization_level": int(
                self.backend_info.details.get("transpile_optimization_level", 1)
            ),
            "transpile_seed": int(self.backend_info.details.get("transpile_seed", int(self.config.seed))),
            "compiled_num_qubits": int(compiled_base.num_qubits),
            "layout_physical_qubits": [int(x) for x in logical_to_physical],
            "requested_slices": [str(x) for x in requested_slices],
        }
        payload: dict[str, Any] = {"shared_compile": shared_compile, "slices": {}}
        for slice_name in requested_slices:
            target, target_details = self._get_backend_scheduled_attribution_target(slice_name)
            try:
                estimate, details = self._evaluate_backend_scheduled_with_target(
                    compiled_base,
                    logical_to_physical,
                    observable,
                    execution_target=(None if slice_name == "full" else target),
                    attribution_slice=str(slice_name),
                    target_details=target_details,
                )
                payload["slices"][str(slice_name)] = {
                    "success": True,
                    "slice": str(slice_name),
                    "components": dict(target_details.get("components", {})),
                    "estimate": estimate,
                    "backend_info": self._snapshot_backend_info(**details),
                    "reason": None,
                    "error": None,
                }
            except Exception as exc:
                payload["slices"][str(slice_name)] = {
                    "success": False,
                    "slice": str(slice_name),
                    "components": dict(target_details.get("components", {})),
                    "estimate": None,
                    "backend_info": self._snapshot_backend_info(**dict(target_details)),
                    "reason": "slice_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return payload

    def _fallback_allowed_for_mode(self) -> bool:
        return (
            str(self.config.noise_mode) in {"shots", "aer_noise"}
            and bool(self.config.allow_aer_fallback)
            and str(self.config.aer_fallback_mode) == "sampler_shots"
        )

    def _can_fallback_from_error(self, exc: Exception) -> bool:
        if not self._fallback_allowed_for_mode():
            return False
        return _looks_like_openmp_shm_abort(str(exc))

    def _activate_sampler_fallback(self, *, reason: str, aer_failed: bool) -> None:
        if self._sampler_fallback is None:
            try:
                from qiskit.primitives import StatevectorSampler
            except Exception as exc:
                raise RuntimeError(
                    "Failed to activate sampler fallback (`StatevectorSampler` unavailable)."
                ) from exc
            self._sampler_fallback = StatevectorSampler(
                default_shots=int(self.config.shots),
                seed=int(self.config.seed),
            )
        self._fallback_reason = str(reason)
        old = self.backend_info
        details = dict(getattr(old, "details", {}))
        details["aer_failed"] = bool(aer_failed)
        details["fallback_used"] = True
        details["fallback_mode"] = str(self.config.aer_fallback_mode)
        details["fallback_reason"] = str(reason)
        details["env_workaround_applied"] = bool(self.config.omp_shm_workaround)
        details["mitigation"] = normalize_mitigation_config(getattr(self.config, "mitigation", "none"))
        details["symmetry_mitigation"] = normalize_symmetry_mitigation_config(
            getattr(self.config, "symmetry_mitigation", "off")
        )
        self.backend_info = NoiseBackendInfo(
            noise_mode=str(self.config.noise_mode),
            estimator_kind="qiskit.primitives.StatevectorSampler(fallback)",
            backend_name=(old.backend_name or "statevector_sampler_fallback"),
            using_fake_backend=bool(old.using_fake_backend),
            details=details,
        )
        self._estimator = None

    def evaluate(
        self,
        circuit: QuantumCircuit,
        observable: SparsePauliOp,
        *,
        runtime_job_observer: Callable[[dict[str, Any]], None] | None = None,
        runtime_trace_context: Mapping[str, Any] | None = None,
    ) -> OracleEstimate:
        if self._closed:
            raise RuntimeError("ExpectationOracle is closed.")

        symmetry_est = self._maybe_evaluate_symmetry_mitigated(circuit, observable)
        if symmetry_est is not None:
            return symmetry_est
        if str(self.config.noise_mode) == "backend_scheduled":
            return self._evaluate_backend_scheduled(circuit, observable)
        if str(self.config.noise_mode) == "runtime":
            return self._evaluate_runtime(
                circuit,
                observable,
                runtime_job_observer=runtime_job_observer,
                runtime_trace_context=runtime_trace_context,
            )

        vals: list[float] = []
        repeats = max(1, int(self.config.oracle_repeats))
        for _ in range(repeats):
            if self._sampler_fallback is not None:
                val = _run_sampler_fallback_job(self._sampler_fallback, circuit, observable)
                vals.append(float(np.real(val)))
                continue
            try:
                execution = _run_estimator_job(self._estimator, circuit, observable)
                vals.append(float(np.real(execution.expectation_value)))
            except Exception as exc:
                if self._can_fallback_from_error(exc):
                    self._activate_sampler_fallback(reason=str(exc), aer_failed=True)
                    val = _run_sampler_fallback_job(self._sampler_fallback, circuit, observable)
                    vals.append(float(np.real(val)))
                else:
                    raise
        return _oracle_estimate_from_samples(vals, aggregate=str(self.config.oracle_aggregate))


_NUMBER_OPERATOR_MATH = "n_p = (I - Z_p) / 2"


def _number_operator_qop(num_qubits: int, index: int) -> SparsePauliOp:
    if index < 0 or index >= int(num_qubits):
        raise ValueError(f"index {index} out of range for num_qubits={num_qubits}")
    chars = ["I"] * int(num_qubits)
    chars[int(num_qubits) - 1 - int(index)] = "Z"
    z_label = "".join(chars)
    return SparsePauliOp.from_list(
        [
            ("I" * int(num_qubits), 0.5),
            (z_label, -0.5),
        ]
    ).simplify(atol=1e-12)


_ALL_Z_FULL_REGISTER_OPERATOR_MATH = "M_Z = Z_(n-1) ... Z_0"


def _all_z_full_register_qop(num_qubits: int) -> SparsePauliOp:
    if int(num_qubits) <= 0:
        raise ValueError("num_qubits must be positive for full-register all-Z diagnostics.")
    return SparsePauliOp.from_list([("Z" * int(num_qubits), 1.0)]).simplify(atol=1e-12)


_DOUBLON_OPERATOR_MATH = "D_i = n_{i,up} n_{i,dn} = (I - Z_up - Z_dn + Z_up Z_dn) / 4"


def _doublon_site_qop(num_qubits: int, up_index: int, dn_index: int) -> SparsePauliOp:
    if up_index == dn_index:
        raise ValueError("up_index and dn_index must differ")
    chars_up = ["I"] * int(num_qubits)
    chars_dn = ["I"] * int(num_qubits)
    chars_both = ["I"] * int(num_qubits)
    chars_up[int(num_qubits) - 1 - int(up_index)] = "Z"
    chars_dn[int(num_qubits) - 1 - int(dn_index)] = "Z"
    chars_both[int(num_qubits) - 1 - int(up_index)] = "Z"
    chars_both[int(num_qubits) - 1 - int(dn_index)] = "Z"
    return SparsePauliOp.from_list(
        [
            ("I" * int(num_qubits), 0.25),
            ("".join(chars_up), -0.25),
            ("".join(chars_dn), -0.25),
            ("".join(chars_both), 0.25),
        ]
    ).simplify(atol=1e-12)


_HH_TOTAL_DOUBLON_OPERATOR_MATH = "D = Σ_i n_{i,up} n_{i,dn}"


def _hh_total_doublon_qop(
    *,
    num_qubits: int,
    num_sites: int,
    ordering: str,
) -> SparsePauliOp:
    if int(num_sites) <= 0:
        raise ValueError("num_sites must be positive for total HH doublon diagnostics.")
    if int(num_qubits) < 2 * int(num_sites):
        raise ValueError("num_qubits must cover the HH fermion register for doublon diagnostics.")
    alpha_indices, beta_indices = _spin_orbital_index_sets(int(num_sites), str(ordering))
    total = SparsePauliOp.from_list([("I" * int(num_qubits), 0.0)]).simplify(atol=1e-12)
    for up_index, dn_index in zip(alpha_indices, beta_indices):
        total = (total + _doublon_site_qop(int(num_qubits), int(up_index), int(dn_index))).simplify(
            atol=1e-12
        )
    return total.simplify(atol=1e-12)


def _ordered_qop_from_exyz(
    ordered_labels_exyz: Sequence[str],
    coeff_map_exyz: dict[str, complex],
    *,
    tol: float = 1e-12,
) -> SparsePauliOp:
    terms: list[tuple[str, complex]] = []
    for lbl in ordered_labels_exyz:
        coeff = complex(coeff_map_exyz[lbl])
        if abs(coeff) <= float(tol):
            continue
        terms.append((_to_ixyz(lbl), coeff))
    if not terms:
        nq = len(ordered_labels_exyz[0]) if ordered_labels_exyz else 1
        terms = [("I" * nq, 0.0 + 0.0j)]
    return SparsePauliOp.from_list(terms).simplify(atol=float(tol))
