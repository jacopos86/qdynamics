"""Paper-I-style resource denominators for AP-McLachlan append ranking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_scoring import (
    Phase1CompileCostOracle,
    SimpleScoreConfig,
    resolve_hardware_cost_lambdas,
)
from pipelines.time_dynamics.ap_mclachlan.support_atoms import SupportAtom

AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1 = "paper_i_proxy_denominator_v1"
AP_APPEND_RANK_SCORE_KIND_V1 = "append_gain_over_paper_i_proxy_denominator_v1"
AP_APPEND_NO_MEASUREMENT_COST_SOURCE_V1 = "ap_append_no_measurement_cache_zero_shot_v1"
AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1 = "family_robust_v1"
AP_APPEND_COST_NORMALIZATION_RAW_LEGACY_V1 = "raw_legacy_v1"
AP_APPEND_COST_FAMILY_SCHEMA_V1 = "ap_append_paper_i_proxy_cost_family_v1"

_COMPONENTS = ("2q", "d", "1q", "theta", "shot")


@dataclass(frozen=True)
class AppendCostSettings:
    """Settings for AP append cost-weighted ranking."""

    cost_model: str = AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1
    cost_normalization_mode: str = AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1
    append_cost_alpha: float = 1.0
    lambda_2q: float = 0.05
    lambda_d: float = 0.05
    lambda_1q: float = 0.025
    lambda_theta: float = 0.0
    lambda_shot: float = 0.02
    scale_floor: float = 1.0e-12

    @classmethod
    def from_config(cls, config: Any) -> "AppendCostSettings":
        return cls(
            cost_model=str(
                getattr(config, "cost_model", AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1)
            ),
            cost_normalization_mode=str(
                getattr(
                    config,
                    "cost_normalization_mode",
                    AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1,
                )
            ),
            append_cost_alpha=float(getattr(config, "append_cost_alpha", 1.0)),
            lambda_2q=float(getattr(config, "append_cost_lambda_2q", 0.05)),
            lambda_d=float(getattr(config, "append_cost_lambda_d", 0.05)),
            lambda_1q=float(getattr(config, "append_cost_lambda_1q", 0.025)),
            lambda_theta=float(getattr(config, "append_cost_lambda_theta", 0.0)),
            lambda_shot=float(getattr(config, "append_cost_lambda_shot", 0.02)),
            scale_floor=float(getattr(config, "append_cost_scale_floor", 1.0e-12)),
        )

    def __post_init__(self) -> None:
        if str(self.cost_model) != AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1:
            raise ValueError(
                "AP append cost_model must be "
                f"{AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1!r}."
            )
        mode = normalize_append_cost_normalization_mode(self.cost_normalization_mode)
        object.__setattr__(self, "cost_normalization_mode", mode)
        for name in (
            "append_cost_alpha",
            "lambda_2q",
            "lambda_d",
            "lambda_1q",
            "lambda_theta",
            "lambda_shot",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not math.isfinite(float(self.scale_floor)) or float(self.scale_floor) <= 0.0:
            raise ValueError("scale_floor must be finite and positive.")

    def paper_i_score_config(self) -> SimpleScoreConfig:
        return SimpleScoreConfig(
            lambda_2q=float(self.lambda_2q),
            lambda_d=float(self.lambda_d),
            lambda_1q=float(self.lambda_1q),
            lambda_theta=float(self.lambda_theta),
            lambda_shot=float(self.lambda_shot),
            hardware_cost_scale_floor=float(self.scale_floor),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "cost_model": str(self.cost_model),
            "cost_normalization_mode": str(self.cost_normalization_mode),
            "append_cost_alpha": float(self.append_cost_alpha),
            "append_cost_lambda_2q": float(self.lambda_2q),
            "append_cost_lambda_d": float(self.lambda_d),
            "append_cost_lambda_1q": float(self.lambda_1q),
            "append_cost_lambda_theta": float(self.lambda_theta),
            "append_cost_lambda_shot": float(self.lambda_shot),
            "append_cost_scale_floor": float(self.scale_floor),
        }


@dataclass(frozen=True)
class AppendCostRawEstimate:
    """Raw Paper-I-style component costs for one AP append atom set."""

    raw_components: Mapping[str, float]
    component_sources: Mapping[str, str]
    inserted_runtime_count: int
    atom_ids: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "raw_components": _component_payload(self.raw_components),
            "component_sources": {str(k): str(v) for k, v in self.component_sources.items()},
            "inserted_runtime_count": int(self.inserted_runtime_count),
            "atom_ids": [str(v) for v in self.atom_ids],
        }


@dataclass(frozen=True)
class AppendCostTelemetry:
    """Cost denominator and rank utility for one AP append atom set."""

    rank_score_kind: str
    cost_model_effective: str
    normalization_mode: str
    raw_components: Mapping[str, float]
    bar_components: Mapping[str, float]
    lambdas: Mapping[str, float]
    lambda_source: str
    hardware_cost_excess_sum: float
    hardware_cost_denominator: float
    append_cost_alpha: float
    utility_denominator: float
    insertion_gain: float | None
    rank_utility: float | None
    component_sources: Mapping[str, str]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "rank_score_kind": str(self.rank_score_kind),
            "cost_model_effective": str(self.cost_model_effective),
            "cost_normalization_mode": str(self.normalization_mode),
            "raw_components": _component_payload(self.raw_components),
            "bar_components": _component_payload(self.bar_components),
            "lambdas": _component_payload(self.lambdas),
            "lambda_source": str(self.lambda_source),
            "hardware_cost_excess_sum": float(self.hardware_cost_excess_sum),
            "hardware_cost_denominator": float(self.hardware_cost_denominator),
            "append_cost_alpha": float(self.append_cost_alpha),
            "utility_denominator": float(self.utility_denominator),
            "insertion_gain": _finite_or_none(self.insertion_gain),
            "rank_utility": _finite_or_none(self.rank_utility),
            "component_sources": {
                str(k): str(v) for k, v in self.component_sources.items()
            },
        }


def normalize_append_cost_normalization_mode(mode: str) -> str:
    value = str(mode or AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1)
    value = value.strip().lower().replace("-", "_")
    aliases = {
        "family": AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1,
        "family_robust": AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1,
        "paper_i_family": AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1,
        "raw": AP_APPEND_COST_NORMALIZATION_RAW_LEGACY_V1,
        "raw_legacy": AP_APPEND_COST_NORMALIZATION_RAW_LEGACY_V1,
        "legacy_raw": AP_APPEND_COST_NORMALIZATION_RAW_LEGACY_V1,
    }
    value = aliases.get(value, value)
    if value not in {
        AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1,
        AP_APPEND_COST_NORMALIZATION_RAW_LEGACY_V1,
    }:
        raise ValueError(
            "unknown AP append cost normalization mode: "
            f"{mode!r}"
        )
    return value


def estimate_append_atom_set_cost(atoms: Sequence[SupportAtom]) -> AppendCostRawEstimate:
    """Estimate raw cost components for one AP append candidate atom set."""

    atom_tuple = tuple(atoms)
    oracle = Phase1CompileCostOracle()
    totals = {key: 0.0 for key in _COMPONENTS}
    sources: dict[str, str] = {
        "2q": "paper_i_phase1_compile_proxy",
        "d": "paper_i_phase1_compile_proxy",
        "1q": "paper_i_phase1_compile_proxy",
        "theta": "ap_inserted_runtime_coordinate_count",
        "shot": AP_APPEND_NO_MEASUREMENT_COST_SOURCE_V1,
    }
    inserted_runtime_count = int(sum(max(0, int(atom.runtime_count)) for atom in atom_tuple))
    for atom in atom_tuple:
        estimate = oracle.estimate(
            candidate_term_count=max(1, int(atom.runtime_count)),
            position_id=0,
            append_position=0,
            refit_active_count=max(0, int(atom.runtime_count)),
            candidate_term=atom.term,
        )
        totals["2q"] += _finite_nonnegative(estimate.c_hat_2q)
        totals["d"] += _finite_nonnegative(estimate.c_hat_d)
        totals["1q"] += _finite_nonnegative(estimate.c_hat_1q)
    totals["theta"] = float(max(0, inserted_runtime_count))
    totals["shot"] = 0.0
    return AppendCostRawEstimate(
        raw_components={key: float(totals[key]) for key in _COMPONENTS},
        component_sources=sources,
        inserted_runtime_count=int(inserted_runtime_count),
        atom_ids=tuple(str(atom.atom_id) for atom in atom_tuple),
    )


def append_cost_telemetry_for_family(
    raw_estimates: Sequence[AppendCostRawEstimate],
    *,
    insertion_gains: Sequence[float | None],
    settings: AppendCostSettings,
) -> tuple[AppendCostTelemetry, ...]:
    """Normalize a candidate family and return cost-weighted append utilities."""

    raw_tuple = tuple(raw_estimates)
    gains = tuple(insertion_gains)
    if len(raw_tuple) != len(gains):
        raise ValueError("raw_estimates and insertion_gains must have the same length.")
    cfg = settings.paper_i_score_config()
    lambdas, lambda_source = resolve_hardware_cost_lambdas(cfg)
    bars_by_index = _bar_components_for_family(raw_tuple, settings=settings)
    out: list[AppendCostTelemetry] = []
    for raw, bars, gain in zip(raw_tuple, bars_by_index, gains):
        excess_sum = float(
            sum(float(lambdas[key]) * float(bars[key]) for key in _COMPONENTS)
        )
        denominator = float(max(1.0, 1.0 + max(0.0, excess_sum)))
        utility_denominator = float(
            max(float(settings.scale_floor), denominator ** float(settings.append_cost_alpha))
        )
        gain_f = _finite_or_none(gain)
        rank_utility = None if gain_f is None else float(gain_f / utility_denominator)
        out.append(
            AppendCostTelemetry(
                rank_score_kind=AP_APPEND_RANK_SCORE_KIND_V1,
                cost_model_effective=AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1,
                normalization_mode=str(settings.cost_normalization_mode),
                raw_components=dict(raw.raw_components),
                bar_components=dict(bars),
                lambdas={str(k): float(v) for k, v in lambdas.items()},
                lambda_source=str(lambda_source),
                hardware_cost_excess_sum=float(max(0.0, excess_sum)),
                hardware_cost_denominator=float(denominator),
                append_cost_alpha=float(settings.append_cost_alpha),
                utility_denominator=float(utility_denominator),
                insertion_gain=gain_f,
                rank_utility=rank_utility,
                component_sources=dict(raw.component_sources),
            )
        )
    return tuple(out)


def _bar_components_for_family(
    raw_estimates: Sequence[AppendCostRawEstimate],
    *,
    settings: AppendCostSettings,
) -> tuple[dict[str, float], ...]:
    if settings.cost_normalization_mode == AP_APPEND_COST_NORMALIZATION_RAW_LEGACY_V1:
        return tuple(
            {key: _finite_nonnegative(raw.raw_components.get(key, 0.0)) for key in _COMPONENTS}
            for raw in raw_estimates
        )
    stats = _family_robust_stats(raw_estimates, scale_floor=float(settings.scale_floor))
    out: list[dict[str, float]] = []
    for raw in raw_estimates:
        bars: dict[str, float] = {}
        for key in _COMPONENTS:
            value = _finite_nonnegative(raw.raw_components.get(key, 0.0))
            median = float(stats["medians"][key])
            scale = max(float(settings.scale_floor), float(stats["scales"][key]))
            bars[key] = float(math.asinh(max(0.0, value - median) / scale))
        out.append(bars)
    return tuple(out)


def _family_robust_stats(
    raw_estimates: Sequence[AppendCostRawEstimate],
    *,
    scale_floor: float,
) -> dict[str, dict[str, float]]:
    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    for key in _COMPONENTS:
        values = [
            _finite_nonnegative(raw.raw_components.get(key, 0.0))
            for raw in raw_estimates
        ]
        median = float(np.median(values)) if values else 0.0
        excesses = [float(value - median) for value in values if value > median]
        scale = float(np.median(excesses)) if excesses else float(scale_floor)
        medians[key] = float(max(0.0, median))
        scales[key] = float(max(float(scale_floor), scale))
    return {"medians": medians, "scales": scales}


def _finite_nonnegative(value: Any, default: float = 0.0) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value_f):
        return float(default)
    return float(max(0.0, value_f))


def _finite_or_none(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return float(value_f)


def _component_payload(values: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(_finite_nonnegative(values.get(key, 0.0))) for key in _COMPONENTS}


__all__ = [
    "AP_APPEND_COST_FAMILY_SCHEMA_V1",
    "AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1",
    "AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1",
    "AP_APPEND_COST_NORMALIZATION_RAW_LEGACY_V1",
    "AP_APPEND_NO_MEASUREMENT_COST_SOURCE_V1",
    "AP_APPEND_RANK_SCORE_KIND_V1",
    "AppendCostRawEstimate",
    "AppendCostSettings",
    "AppendCostTelemetry",
    "append_cost_telemetry_for_family",
    "estimate_append_atom_set_cost",
    "normalize_append_cost_normalization_mode",
]
