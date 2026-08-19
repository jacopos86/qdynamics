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




def estimate_prune_atom_set_cost(atoms: Sequence[SupportAtom]) -> AppendCostRawEstimate:
    """Estimate saved resource cost for deleting an active atom set."""

    return estimate_append_atom_set_cost(tuple(atoms))




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
            "prune_condition_lambda_kappa_dam",
        )
    )


__all__ = [
    "AP_PRUNE_RANK_SCORE_KIND_V1",
    "AP_PRUNE_CONDITIONED_RANK_SCORE_KIND_V1",
    "PruneCostSettings",
    "estimate_prune_atom_set_cost",
]
