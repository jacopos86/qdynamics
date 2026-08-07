"""Paper-I-style resource pressure for AP-McLachlan prune ranking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.ap_mclachlan.append_cost import (
    AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1,
    AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1,
    AppendCostRawEstimate,
    AppendCostSettings,
    append_cost_telemetry_for_family,
    estimate_append_atom_set_cost,
)
from pipelines.time_dynamics.ap_mclachlan.support_atoms import SupportAtom

AP_PRUNE_RANK_SCORE_KIND_V1 = "prune_cost_pressure_over_loss_history_v1"
AP_PRUNE_CONDITIONED_RANK_SCORE_KIND_V1 = (
    "prune_cost_pressure_conditioning_over_loss_history_v1"
)


@dataclass(frozen=True)
class PruneCostSettings:
    """Settings for Paper-II active support-patch prune ranking."""

    cost_model: str = AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1
    cost_normalization_mode: str = AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1
    prune_cost_alpha: float = 1.0
    prune_history_lambda: float = 1.0
    prune_condition_lambda_kappa_rel: float = 0.0
    prune_condition_lambda_schur: float = 0.0
    prune_condition_lambda_kappa_hist: float = 0.0
    prune_condition_lambda_kappa_dam: float = 0.0
    eps_loss: float = 1.0e-14
    append_cost_lambda_2q: float = 0.05
    append_cost_lambda_d: float = 0.05
    append_cost_lambda_1q: float = 0.025
    append_cost_lambda_theta: float = 0.0
    append_cost_lambda_shot: float = 0.02
    append_cost_scale_floor: float = 1.0e-12

    @classmethod
    def from_config(cls, config: Any) -> "PruneCostSettings":
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
            prune_cost_alpha=float(getattr(config, "prune_cost_alpha", 1.0)),
            prune_history_lambda=float(getattr(config, "prune_history_lambda", 1.0)),
            prune_condition_lambda_kappa_rel=float(
                getattr(config, "prune_condition_lambda_kappa_rel", 0.0)
            ),
            prune_condition_lambda_schur=float(
                getattr(config, "prune_condition_lambda_schur", 0.0)
            ),
            prune_condition_lambda_kappa_hist=float(
                getattr(config, "prune_condition_lambda_kappa_hist", 0.0)
            ),
            prune_condition_lambda_kappa_dam=float(
                getattr(config, "prune_condition_lambda_kappa_dam", 0.0)
            ),
            eps_loss=float(getattr(config, "eps_loss", 1.0e-14)),
            append_cost_lambda_2q=float(getattr(config, "append_cost_lambda_2q", 0.05)),
            append_cost_lambda_d=float(getattr(config, "append_cost_lambda_d", 0.05)),
            append_cost_lambda_1q=float(getattr(config, "append_cost_lambda_1q", 0.025)),
            append_cost_lambda_theta=float(
                getattr(config, "append_cost_lambda_theta", 0.0)
            ),
            append_cost_lambda_shot=float(getattr(config, "append_cost_lambda_shot", 0.02)),
            append_cost_scale_floor=float(
                getattr(config, "append_cost_scale_floor", 1.0e-12)
            ),
        )

    def __post_init__(self) -> None:
        AppendCostSettings(
            cost_model=str(self.cost_model),
            cost_normalization_mode=str(self.cost_normalization_mode),
            append_cost_alpha=1.0,
            lambda_2q=float(self.append_cost_lambda_2q),
            lambda_d=float(self.append_cost_lambda_d),
            lambda_1q=float(self.append_cost_lambda_1q),
            lambda_theta=float(self.append_cost_lambda_theta),
            lambda_shot=float(self.append_cost_lambda_shot),
            scale_floor=float(self.append_cost_scale_floor),
        )
        for name in (
            "prune_cost_alpha",
            "prune_history_lambda",
            "prune_condition_lambda_kappa_rel",
            "prune_condition_lambda_schur",
            "prune_condition_lambda_kappa_hist",
            "prune_condition_lambda_kappa_dam",
            "eps_loss",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

    def append_denominator_settings(self) -> AppendCostSettings:
        return AppendCostSettings(
            cost_model=str(self.cost_model),
            cost_normalization_mode=str(self.cost_normalization_mode),
            append_cost_alpha=1.0,
            lambda_2q=float(self.append_cost_lambda_2q),
            lambda_d=float(self.append_cost_lambda_d),
            lambda_1q=float(self.append_cost_lambda_1q),
            lambda_theta=float(self.append_cost_lambda_theta),
            lambda_shot=float(self.append_cost_lambda_shot),
            scale_floor=float(self.append_cost_scale_floor),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "cost_model": str(self.cost_model),
            "cost_normalization_mode": str(self.cost_normalization_mode),
            "prune_cost_alpha": float(self.prune_cost_alpha),
            "prune_history_lambda": float(self.prune_history_lambda),
            "prune_condition_lambda_kappa_rel": float(
                self.prune_condition_lambda_kappa_rel
            ),
            "prune_condition_lambda_schur": float(
                self.prune_condition_lambda_schur
            ),
            "prune_condition_lambda_kappa_hist": float(
                self.prune_condition_lambda_kappa_hist
            ),
            "prune_condition_lambda_kappa_dam": float(
                self.prune_condition_lambda_kappa_dam
            ),
            "eps_loss": float(self.eps_loss),
            "append_cost_lambda_2q": float(self.append_cost_lambda_2q),
            "append_cost_lambda_d": float(self.append_cost_lambda_d),
            "append_cost_lambda_1q": float(self.append_cost_lambda_1q),
            "append_cost_lambda_theta": float(self.append_cost_lambda_theta),
            "append_cost_lambda_shot": float(self.append_cost_lambda_shot),
            "append_cost_scale_floor": float(self.append_cost_scale_floor),
        }


@dataclass(frozen=True)
class PruneCostTelemetry:
    """Cost-pressure ranking telemetry for one prune candidate batch."""

    rank_score_kind: str
    cost_model_effective: str
    normalization_mode: str
    raw_components: Mapping[str, float]
    bar_components: Mapping[str, float]
    lambdas: Mapping[str, float]
    lambda_source: str
    hardware_cost_excess_sum: float
    hardware_cost_denominator: float
    prune_cost_alpha: float
    saved_cost_pressure: float
    deletion_loss: float | None
    historical_deletion_loss: float
    history_count: int
    prune_history_lambda: float
    conditioning_components: Mapping[str, float]
    conditioning_lambdas: Mapping[str, float]
    conditioning_pressure_multiplier: float
    conditioning_damage_penalty: float
    eps_loss: float
    utility_denominator: float
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
            "prune_cost_alpha": float(self.prune_cost_alpha),
            "saved_cost_pressure": float(self.saved_cost_pressure),
            "deletion_loss": _finite_or_none(self.deletion_loss),
            "historical_deletion_loss": float(self.historical_deletion_loss),
            "history_count": int(self.history_count),
            "prune_history_lambda": float(self.prune_history_lambda),
            "conditioning_components": _conditioning_payload(
                self.conditioning_components
            ),
            "conditioning_lambdas": _conditioning_lambda_payload(
                self.conditioning_lambdas
            ),
            "conditioning_pressure_multiplier": float(
                self.conditioning_pressure_multiplier
            ),
            "conditioning_damage_penalty": float(self.conditioning_damage_penalty),
            "eps_loss": float(self.eps_loss),
            "utility_denominator": float(self.utility_denominator),
            "rank_utility": _finite_or_none(self.rank_utility),
            "component_sources": {
                str(k): str(v) for k, v in self.component_sources.items()
            },
        }


def estimate_prune_atom_set_cost(atoms: Sequence[SupportAtom]) -> AppendCostRawEstimate:
    """Estimate saved resource cost for deleting an active atom set."""

    return estimate_append_atom_set_cost(tuple(atoms))


def prune_cost_telemetry_for_family(
    raw_estimates: Sequence[AppendCostRawEstimate],
    *,
    deletion_losses: Sequence[float | None],
    historical_losses: Sequence[float],
    history_counts: Sequence[int],
    conditioning_components: Sequence[Mapping[str, Any] | None] | None = None,
    settings: PruneCostSettings,
) -> tuple[PruneCostTelemetry, ...]:
    """Normalize a prune candidate family and rank saved cost over loss."""

    raw_tuple = tuple(raw_estimates)
    losses = tuple(deletion_losses)
    histories = tuple(historical_losses)
    counts = tuple(history_counts)
    conditioning = (
        tuple({} for _ in raw_tuple)
        if conditioning_components is None
        else tuple(conditioning_components)
    )
    if not (
        len(raw_tuple)
        == len(losses)
        == len(histories)
        == len(counts)
        == len(conditioning)
    ):
        raise ValueError(
            "raw_estimates, deletion_losses, historical_losses, history_counts, "
            "and conditioning_components must have the same length."
        )
    append_costs = append_cost_telemetry_for_family(
        raw_tuple,
        insertion_gains=(1.0 for _ in raw_tuple),
        settings=settings.append_denominator_settings(),
    )
    out: list[PruneCostTelemetry] = []
    conditioned = _conditioning_enabled(settings)
    for raw, append_cost, loss, hist, count, cond_raw in zip(
        raw_tuple,
        append_costs,
        losses,
        histories,
        counts,
        conditioning,
    ):
        deletion_loss = _finite_or_none(loss)
        hist_loss = max(0.0, _finite_or_none(hist) or 0.0)
        cond = _conditioning_payload(cond_raw or {})
        conditioning_multiplier = float(
            1.0
            + float(settings.prune_condition_lambda_kappa_rel)
            * float(cond["d_kappa_rel"])
            + float(settings.prune_condition_lambda_schur) * float(cond["d_schur"])
            + float(settings.prune_condition_lambda_kappa_hist)
            * float(cond["d_kappa_schur_hist"])
        )
        conditioning_multiplier = max(1.0, conditioning_multiplier)
        conditioning_damage = float(
            float(settings.prune_condition_lambda_kappa_dam)
            * float(cond["d_kappa_dam"])
        )
        denominator = float(
            max(
                float(settings.eps_loss),
                (0.0 if deletion_loss is None else float(deletion_loss))
                + float(settings.prune_history_lambda) * hist_loss
                + conditioning_damage
                + float(settings.eps_loss),
            )
        )
        saved_pressure = float(
            max(float(settings.append_cost_scale_floor), append_cost.hardware_cost_denominator)
            ** float(settings.prune_cost_alpha)
        )
        rank = (
            None
            if deletion_loss is None
            else float(saved_pressure * conditioning_multiplier / denominator)
        )
        out.append(
            PruneCostTelemetry(
                rank_score_kind=(
                    AP_PRUNE_CONDITIONED_RANK_SCORE_KIND_V1
                    if conditioned
                    else AP_PRUNE_RANK_SCORE_KIND_V1
                ),
                cost_model_effective=AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1,
                normalization_mode=str(append_cost.normalization_mode),
                raw_components=dict(raw.raw_components),
                bar_components=dict(append_cost.bar_components),
                lambdas=dict(append_cost.lambdas),
                lambda_source=str(append_cost.lambda_source),
                hardware_cost_excess_sum=float(append_cost.hardware_cost_excess_sum),
                hardware_cost_denominator=float(append_cost.hardware_cost_denominator),
                prune_cost_alpha=float(settings.prune_cost_alpha),
                saved_cost_pressure=float(saved_pressure),
                deletion_loss=deletion_loss,
                historical_deletion_loss=float(hist_loss),
                history_count=int(count),
                prune_history_lambda=float(settings.prune_history_lambda),
                conditioning_components=cond,
                conditioning_lambdas=_conditioning_lambda_payload(
                    {
                        "kappa_rel": settings.prune_condition_lambda_kappa_rel,
                        "schur": settings.prune_condition_lambda_schur,
                        "kappa_hist": settings.prune_condition_lambda_kappa_hist,
                        "kappa_dam": settings.prune_condition_lambda_kappa_dam,
                    }
                ),
                conditioning_pressure_multiplier=float(conditioning_multiplier),
                conditioning_damage_penalty=float(conditioning_damage),
                eps_loss=float(settings.eps_loss),
                utility_denominator=float(denominator),
                rank_utility=rank,
                component_sources=dict(raw.component_sources),
            )
        )
    return tuple(out)


def _finite_or_none(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return float(value_f)


def _component_payload(values: Mapping[str, Any]) -> dict[str, float]:
    keys = ("2q", "d", "1q", "theta", "shot")
    return {key: float(max(0.0, _finite_or_none(values.get(key, 0.0)) or 0.0)) for key in keys}


def _conditioning_payload(values: Mapping[str, Any]) -> dict[str, float]:
    keys = ("d_kappa_rel", "d_schur", "d_kappa_schur_hist", "d_kappa_dam")
    return {
        key: float(max(0.0, _finite_or_none(values.get(key, 0.0)) or 0.0))
        for key in keys
    }


def _conditioning_lambda_payload(values: Mapping[str, Any]) -> dict[str, float]:
    keys = ("kappa_rel", "schur", "kappa_hist", "kappa_dam")
    return {
        key: float(max(0.0, _finite_or_none(values.get(key, 0.0)) or 0.0))
        for key in keys
    }


def _conditioning_enabled(settings: PruneCostSettings) -> bool:
    return any(
        float(getattr(settings, name)) > 0.0
        for name in (
            "prune_condition_lambda_kappa_rel",
            "prune_condition_lambda_schur",
            "prune_condition_lambda_kappa_hist",
            "prune_condition_lambda_kappa_dam",
        )
    )


__all__ = [
    "AP_PRUNE_RANK_SCORE_KIND_V1",
    "AP_PRUNE_CONDITIONED_RANK_SCORE_KIND_V1",
    "PruneCostSettings",
    "PruneCostTelemetry",
    "estimate_prune_atom_set_cost",
    "prune_cost_telemetry_for_family",
]
