#!/usr/bin/env python3
"""Noise/runtime expectation oracle utilities for HH/Hubbard validation.

This module stays in wrapper/benchmark space. It does not modify core operator
algebra modules and only adapts existing PauliPolynomial + ansatz objects to
Qiskit primitives at the boundary.
"""

from __future__ import annotations

import json
import os
import inspect
import subprocess
import sys
import copy
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
)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from src.quantum.ansatz_parameterization import AnsatzParameterLayout


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
        )
        self._backend_target = None
        self._compiled_base_cache: dict[int, dict[str, Any]] = {}
        self._backend_scheduled_repeat_base_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self._backend_scheduled_attribution_targets: dict[str, dict[str, Any]] = {}
        self._backend_scheduled_noise_model = None
        self._backend_scheduled_local_dd_retry_target = None
        self._backend_scheduled_local_dd_retry_details: dict[str, Any] | None = None
        self._backend_scheduled_local_dd_retry_engaged = False
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
            return cached
        if self._backend_target is None:
            raise RuntimeError("backend_scheduled backend target is unavailable.")
        transpile_seed = int(self.config.seed if self.config.seed_transpiler is None else self.config.seed_transpiler)
        optimization_level = int(self.config.transpile_optimization_level)
        cached = compile_circuit_for_local_backend(
            circuit,
            self._backend_target,
            seed_transpiler=transpile_seed,
            optimization_level=optimization_level,
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
            return cached
        if self._backend_target is None:
            raise RuntimeError("runtime backend target is unavailable.")
        transpile_seed = int(self.config.seed if self.config.seed_transpiler is None else self.config.seed_transpiler)
        optimization_level = int(self.config.transpile_optimization_level)
        cached = compile_circuit_for_local_backend(
            circuit,
            self._backend_target,
            seed_transpiler=transpile_seed,
            optimization_level=optimization_level,
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

    def _ensure_mthree_calibration(self, active_physical_qubits: Sequence[int]) -> None:
        qubits = tuple(int(q) for q in active_physical_qubits)
        if qubits in self._mthree_calibrated_qubits:
            return
        if self._backend_target is None:
            raise RuntimeError("backend_scheduled backend target is unavailable.")
        if self._mthree_mitigator is None:
            if self._mthree_module is None:
                self._mthree_module = _resolve_mthree()
            self._mthree_mitigator = self._mthree_module.M3Mitigation(self._backend_target)
        self._mthree_mitigator.cals_from_system(
            qubits=list(qubits),
            shots=int(self.config.shots),
            async_cal=False,
        )
        self._mthree_calibrated_qubits.add(qubits)

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
            repeat_base, twirling_details = self._get_backend_scheduled_repeat_base(
                compiled_base,
                repeat_idx=int(rep),
            )
            last_twirling_details = dict(twirling_details)
            total = 0.0 + 0.0j
            for label, coeff in observable.to_list():
                coeff_c = complex(coeff)
                label_s = str(label).upper()
                if all(ch == "I" for ch in label_s):
                    total += coeff_c
                    continue
                counts, active_physical, term_details = self._backend_scheduled_term_counts(
                    repeat_base,
                    logical_to_physical,
                    label_s,
                    repeat_idx=int(rep),
                    execution_target=execution_target,
                )
                last_dd_details = dict(term_details.get("local_dynamical_decoupling", {}))
                if mitigation_mode == "readout":
                    self._ensure_mthree_calibration(active_physical)
                    quasi, mitigation_details = _apply_mthree_readout_correction(
                        mitigator=self._mthree_mitigator,
                        counts=counts,
                        active_physical_qubits=active_physical,
                    )
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
        }
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
