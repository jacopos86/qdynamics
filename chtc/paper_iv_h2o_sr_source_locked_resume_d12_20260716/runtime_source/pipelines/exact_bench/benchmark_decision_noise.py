#!/usr/bin/env python3
"""Benchmark decision/evaluation-time value-noise helpers.

This module owns the distinct ``benchmark_decision_noise_*`` namespace used by
exact-bench comparator rows.  The semantic is intentionally separate from the
existing post-result ``benchmark_value_noise_*`` overlay and from Phase3/SNAKE
``phase3_oracle_value_noise_*`` controller noise.

Foundation-slice behavior is conservative: helpers can parse/record deterministic
Gaussian draws, but dispatch surfaces may still fail closed until a true local
runner seam is wired and tested.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping, MutableMapping

BENCHMARK_DECISION_NOISE_SEMANTIC = "benchmark_decision_value_noise_not_physical_shots_v1"
BENCHMARK_DECISION_NOISE_TSV_FIELDS = (
    "benchmark_decision_noise_model",
    "benchmark_decision_noise_std",
    "benchmark_decision_noise_seed",
)
BENCHMARK_DECISION_NOISE_MODEL_CHOICES = ("off", "gaussian_iid_v1")
BENCHMARK_DECISION_NOISE_MODEL_CHOICE_SET = frozenset(BENCHMARK_DECISION_NOISE_MODEL_CHOICES)
BENCHMARK_DECISION_NOISE_ENV_PREFIX = "GENERIC_STATIC_TABLE_"
ALGORITHMIC_MEASUREMENT_WORK_SCHEMA = "algorithmic_measurement_work_v1"


@dataclass(frozen=True)
class BenchmarkDecisionNoiseConfig:
    """Validated run input for benchmark decision/evaluation-time noise."""

    enabled: bool
    model: str
    std: float
    seed: int | None
    seed_source: str
    semantic: str = BENCHMARK_DECISION_NOISE_SEMANTIC

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def disabled_config(*, seed_source: str = "omitted") -> BenchmarkDecisionNoiseConfig:
    return BenchmarkDecisionNoiseConfig(
        enabled=False,
        model="off",
        std=0.0,
        seed=None,
        seed_source=str(seed_source),
    )


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default, ensure_ascii=True)


def stable_hash_int(*parts: Any, bits: int = 63) -> int:
    """Return a stable non-negative integer derived from JSON-normalized parts."""

    blob = _stable_json([str(part) for part in parts]).encode("utf-8")
    digest = hashlib.sha256(blob).digest()
    value = int.from_bytes(digest, "big")
    return int(value & ((1 << int(bits)) - 1))


def _env_candidates(field: str) -> tuple[str, ...]:
    key = str(field)
    return (key, key.upper(), f"{BENCHMARK_DECISION_NOISE_ENV_PREFIX}{key.upper()}")


def _env_style_value(values: Mapping[str, Any], field: str) -> str | None:
    for key in _env_candidates(field):
        raw = values.get(key)
        if raw not in {None, ""}:
            return str(raw).strip()
    return None


def _parse_float(raw: str, *, field: str) -> float:
    try:
        value = float(str(raw).strip())
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"{field} must be finite numeric when provided; got {raw!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite numeric when provided; got {raw!r}.")
    return float(value)


def _parse_seed(raw: str, *, field: str) -> int:
    try:
        return int(str(raw).strip(), 10)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"{field} must be an integer seed when provided; got {raw!r}.") from exc


def decision_noise_requested_from_env_values(values: Mapping[str, Any]) -> bool:
    """Return whether env/TSV-style values request decision noise.

    This intentionally treats malformed non-empty numeric strings as requested so
    the caller can surface a validation error instead of silently ignoring them.
    """

    model = str(_env_style_value(values, "benchmark_decision_noise_model") or "off").strip().lower()
    if model != "off":
        return True
    raw_std = _env_style_value(values, "benchmark_decision_noise_std")
    if raw_std not in {None, ""}:
        try:
            if float(str(raw_std)) != 0.0:
                return True
        except Exception:  # noqa: BLE001 - malformed input is still a request
            return True
    return _env_style_value(values, "benchmark_decision_noise_seed") not in {None, ""}


def config_from_env_values(
    values: Mapping[str, Any] | None,
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
) -> BenchmarkDecisionNoiseConfig:
    """Parse and validate ``benchmark_decision_noise_*`` env/TSV values."""

    raw_values = values or {}
    raw = {field: _env_style_value(raw_values, field) for field in BENCHMARK_DECISION_NOISE_TSV_FIELDS}
    model = str(raw["benchmark_decision_noise_model"] or "off").strip().lower() or "off"
    if model not in BENCHMARK_DECISION_NOISE_MODEL_CHOICE_SET:
        raise ValueError(
            "benchmark_decision_noise_model must be one of "
            f"{sorted(BENCHMARK_DECISION_NOISE_MODEL_CHOICE_SET)}."
        )

    std = 0.0
    if raw["benchmark_decision_noise_std"] not in {None, ""}:
        std = _parse_float(
            str(raw["benchmark_decision_noise_std"]),
            field="benchmark_decision_noise_std",
        )

    seed: int | None = None
    seed_source = "omitted"
    if raw["benchmark_decision_noise_seed"] not in {None, ""}:
        seed = _parse_seed(
            str(raw["benchmark_decision_noise_seed"]),
            field="benchmark_decision_noise_seed",
        )
        seed_source = "env"

    if model == "off":
        if seed is not None:
            raise ValueError(
                "benchmark_decision_noise_seed requires benchmark_decision_noise_model='gaussian_iid_v1'."
            )
        if std != 0.0:
            raise ValueError("benchmark_decision_noise_model='off' requires benchmark_decision_noise_std == 0.")
        return disabled_config(seed_source=seed_source)

    if model == "gaussian_iid_v1":
        if (not math.isfinite(std)) or std <= 0.0:
            raise ValueError(
                "benchmark_decision_noise_model='gaussian_iid_v1' requires finite "
                "benchmark_decision_noise_std > 0."
            )
        if seed is None:
            seed = stable_hash_int(
                BENCHMARK_DECISION_NOISE_SEMANTIC,
                family,
                case_id,
                algorithm_id,
                model,
                repr(float(std)),
            )
            seed_source = "derived_stable_hash_v1"
        return BenchmarkDecisionNoiseConfig(
            enabled=True,
            model=model,
            std=float(std),
            seed=int(seed),
            seed_source=seed_source,
        )

    raise ValueError(f"Unsupported benchmark_decision_noise_model {model!r}.")


def config_from_env(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    env: Mapping[str, Any] | None = None,
) -> BenchmarkDecisionNoiseConfig:
    return config_from_env_values(
        os.environ if env is None else env,
        family=family,
        case_id=case_id,
        algorithm_id=algorithm_id,
    )


def coerce_config(
    value: BenchmarkDecisionNoiseConfig | Mapping[str, Any] | None,
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
) -> BenchmarkDecisionNoiseConfig:
    if value is None:
        return disabled_config()
    if isinstance(value, BenchmarkDecisionNoiseConfig):
        return value
    if isinstance(value, Mapping):
        # Accept either validated metadata keys or env-style field names.
        if "enabled" in value and "model" in value:
            enabled = bool(value.get("enabled", False))
            model = str(value.get("model") or "off").strip().lower()
            if not enabled and model == "off":
                return disabled_config(seed_source=str(value.get("seed_source") or "omitted"))
            env_values = {
                "benchmark_decision_noise_model": model,
                "benchmark_decision_noise_std": value.get("std", ""),
                "benchmark_decision_noise_seed": "" if value.get("seed") is None else value.get("seed"),
            }
            return config_from_env_values(
                env_values,
                family=family,
                case_id=case_id,
                algorithm_id=algorithm_id,
            )
        return config_from_env_values(value, family=family, case_id=case_id, algorithm_id=algorithm_id)
    raise TypeError(f"Unsupported benchmark decision-noise config type: {type(value).__name__}")


def deterministic_standard_normal(*, seed: int, scope: Mapping[str, Any]) -> float:
    """Deterministically draw a standard normal for a seed/scope pair."""

    material = _stable_json(
        {
            "semantic": BENCHMARK_DECISION_NOISE_SEMANTIC,
            "model": "gaussian_iid_v1",
            "seed": int(seed),
            "scope": dict(scope),
        }
    ).encode("utf-8")

    def _uniform(label: bytes) -> float:
        digest = hashlib.sha256(material + b"|" + label).digest()
        mantissa = int.from_bytes(digest[:8], "big") >> 11
        return (float(mantissa) + 0.5) / float(1 << 53)

    u1 = _uniform(b"u1")
    u2 = _uniform(b"u2")
    return float(math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2))


def deterministic_gaussian_draw(config: BenchmarkDecisionNoiseConfig, *, scope: Mapping[str, Any]) -> float:
    if not bool(config.enabled):
        return 0.0
    if config.model != "gaussian_iid_v1" or config.seed is None:
        raise ValueError("deterministic_gaussian_draw requires enabled gaussian_iid_v1 config with a seed.")
    return float(float(config.std) * deterministic_standard_normal(seed=int(config.seed), scope=scope))


def _finite_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"decision-noise value must be finite; got {value!r}")
    return float(out)


def _metadata_common(config: BenchmarkDecisionNoiseConfig) -> dict[str, Any]:
    return {
        "enabled": bool(config.enabled),
        "model": str(config.model),
        "std": float(config.std),
        "seed": None if config.seed is None else int(config.seed),
        "seed_source": str(config.seed_source),
        "semantic": BENCHMARK_DECISION_NOISE_SEMANTIC,
        "physical_shots_unchanged": True,
        "algorithmic_measurement_work_schema": ALGORITHMIC_MEASUREMENT_WORK_SCHEMA,
        "algorithmic_measurement_work_unchanged": True,
    }


class BenchmarkDecisionNoiseRecorder:
    """Record deterministic decision-noise draws for one runner invocation."""

    def __init__(
        self,
        config: BenchmarkDecisionNoiseConfig,
        *,
        base_scope: Mapping[str, Any] | None = None,
        trace_limit: int = 32,
    ) -> None:
        self.config = config
        self.base_scope = dict(base_scope or {})
        self.trace_limit = int(trace_limit)
        self.events: list[dict[str, Any]] = []
        self.draw_count_by_surface: dict[str, int] = {}
        self._draw_count_total = 0
        self._trace_truncated_count = 0

    @property
    def draw_count_total(self) -> int:
        return int(self._draw_count_total)

    def apply(
        self,
        value: float,
        *,
        surface: str,
        value_kind: str,
        phase: str,
        extra_scope: Mapping[str, Any] | None = None,
    ) -> float:
        ideal_value = _finite_float(value)
        if not bool(self.config.enabled):
            return ideal_value
        event_index = int(self._draw_count_total)
        surface_key = str(surface)
        scope = {
            "base_scope": dict(self.base_scope),
            "surface": surface_key,
            "value_kind": str(value_kind),
            "phase": str(phase),
            "event_index": event_index,
        }
        if extra_scope:
            scope["extra_scope"] = dict(extra_scope)
        noise_draw = deterministic_gaussian_draw(self.config, scope=scope)
        decision_value = float(ideal_value + noise_draw)
        self._draw_count_total += 1
        self.draw_count_by_surface[surface_key] = int(self.draw_count_by_surface.get(surface_key, 0) + 1)
        event = {
            "event_index": event_index,
            "surface": surface_key,
            "value_kind": str(value_kind),
            "phase": str(phase),
            "value_ideal": ideal_value,
            "noise_draw": noise_draw,
            "value_decision": decision_value,
            "scope": scope,
        }
        if len(self.events) < self.trace_limit:
            self.events.append(event)
        else:
            self._trace_truncated_count += 1
        return decision_value

    def summary(
        self,
        *,
        status: str = "ok",
        supported: bool = True,
        applied: bool | None = None,
        reason: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        applied_value = bool(self.config.enabled and supported and self.draw_count_total > 0) if applied is None else bool(applied)
        metadata: dict[str, Any] = {
            **_metadata_common(self.config),
            "status": str(status),
            "supported": bool(supported),
            "applied": applied_value,
            "draw_count_total": int(self.draw_count_total),
            "draw_count_by_surface": dict(sorted(self.draw_count_by_surface.items())),
            "surfaces_affected": sorted(self.draw_count_by_surface),
            "trace_preview": list(self.events),
            "trace_truncated_count": int(self._trace_truncated_count),
            "scope": dict(self.base_scope),
        }
        if reason not in {None, ""}:
            metadata["reason"] = str(reason)
        if extra:
            metadata.update(dict(extra))
        return metadata


def metadata_not_requested(
    *,
    family: str | None = None,
    case_id: str | None = None,
    algorithm_id: str | None = None,
) -> dict[str, Any]:
    scope = {
        key: value
        for key, value in {
            "family": family,
            "case_id": case_id,
            "algorithm_id": algorithm_id,
        }.items()
        if value not in {None, ""}
    }
    return {
        **_metadata_common(disabled_config()),
        "status": "not_requested",
        "supported": None,
        "applied": False,
        "draw_count_total": 0,
        "draw_count_by_surface": {},
        "surfaces_affected": [],
        "trace_preview": [],
        "trace_truncated_count": 0,
        "scope": scope,
    }


def unsupported_metadata(
    config: BenchmarkDecisionNoiseConfig,
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    dispatch: str | None,
    reason: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        **_metadata_common(config),
        "status": "unsupported",
        "supported": False,
        "applied": False,
        "fail_closed": True,
        "reason": str(reason),
        "dispatch": None if dispatch is None else str(dispatch),
        "draw_count_total": 0,
        "draw_count_by_surface": {},
        "surfaces_affected": [],
        "trace_preview": [],
        "trace_truncated_count": 0,
        "scope": {
            "family": str(family),
            "case_id": str(case_id),
            "algorithm_id": str(algorithm_id),
        },
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


def copy_decision_noise_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly copy for row/payload fan-out."""

    return json.loads(json.dumps(dict(metadata), sort_keys=True, default=_json_default))


__all__ = [
    "ALGORITHMIC_MEASUREMENT_WORK_SCHEMA",
    "BENCHMARK_DECISION_NOISE_ENV_PREFIX",
    "BENCHMARK_DECISION_NOISE_MODEL_CHOICES",
    "BENCHMARK_DECISION_NOISE_MODEL_CHOICE_SET",
    "BENCHMARK_DECISION_NOISE_SEMANTIC",
    "BENCHMARK_DECISION_NOISE_TSV_FIELDS",
    "BenchmarkDecisionNoiseConfig",
    "BenchmarkDecisionNoiseRecorder",
    "coerce_config",
    "config_from_env",
    "config_from_env_values",
    "copy_decision_noise_metadata",
    "decision_noise_requested_from_env_values",
    "deterministic_gaussian_draw",
    "deterministic_standard_normal",
    "disabled_config",
    "metadata_not_requested",
    "stable_hash_int",
    "unsupported_metadata",
]
