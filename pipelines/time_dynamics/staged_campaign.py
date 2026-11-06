"""Pure orchestration policy for the staged Paper-II accuracy campaign.

This module decides *which registered run should happen next*.  It does not
implement APM, AVQDS, exact propagation, or a time-step controller.  Run
commands are always resolved through :mod:`pipelines.time_dynamics.paper_ii_runs`.

The campaign is deliberately replayable: a ledger is an immutable sequence of
scientific rung records, and :meth:`StagedCampaign.next_actions` is a pure
function of that ledger.  Restarting an interrupted executor therefore emits
the unfinished actions instead of rebuilding completed trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.response_scales import (
    is_response_scale_index,
    response_scale_row,
    validate_response_scale_index,
)

from pipelines.time_dynamics import paper_ii_runs as _runs


METHOD_APM = "exchange"
METHOD_AVQDS = "avqds"

STATUS_ACCURACY_QUALIFIED = "accuracy_qualified"
STATUS_REQUIRES_THRESHOLD_EXTENSION = "requires_threshold_extension"
STATUS_REQUIRES_THRESHOLD_COARSENING = "requires_threshold_coarsening"
STATUS_SAFEGUARD_TRIPPED = "safeguard_tripped"
RUNG_STATUSES = frozenset(
    {
        STATUS_ACCURACY_QUALIFIED,
        STATUS_REQUIRES_THRESHOLD_EXTENSION,
        STATUS_REQUIRES_THRESHOLD_COARSENING,
        STATUS_SAFEGUARD_TRIPPED,
    }
)

POLICY_COMPONENTWISE_CEILING = "componentwise_ceiling"
POLICY_MEAN_INFIDELITY_WINDOW = "mean_infidelity_window"
POLICY_RESPONSE_NORMALIZED_MATCHED = "response_normalized_matched_pair"

STAGE_SCREEN_AVQDS = "screen_avqds_t5"
STAGE_PROMOTE_AVQDS = "promote_avqds_t20"
STAGE_SCREEN_APM = "screen_apm_t5"
STAGE_PROMOTE_APM = "promote_apm_t20"
STAGE_SCREEN_T5_FRONTIER = "screen_t5_frontier"
STAGE_PROMOTE_T20_FRONTIER = "promote_t20_frontier"
STAGE_ACTIVE_FRONTIER = "active_t5_t20_frontier"
STAGE_COMPLETE = "complete"

ACTION_RUN = "run"
ACTION_RECORD_SAFEGUARD = "record_safeguard"

LEDGER_SCHEMA_V1 = "paper_ii_staged_campaign_ledger_v1"


def _finite_nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _canonical_threshold(value: float) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("activation threshold must be finite and positive")
    return float(f"{threshold:.12e}")


def _same_threshold(left: float, right: float) -> bool:
    """Treat roundoff-equivalent factor-three rungs as one identity."""

    return math.isclose(
        _canonical_threshold(left),
        _canonical_threshold(right),
        rel_tol=5.0e-12,
        abs_tol=0.0,
    )


@dataclass(frozen=True, order=True)
class CellKey:
    """One method/controller/drive configuration for one physical regime."""

    regime: str
    drive: str
    method: str
    controller: str

    def __post_init__(self) -> None:
        if not all(
            str(value).strip()
            for value in (self.regime, self.drive, self.method, self.controller)
        ):
            raise ValueError("cell-key fields must be non-empty")
        if self.method not in (METHOD_AVQDS, METHOD_APM):
            raise ValueError(f"unsupported staged-campaign method: {self.method!r}")

    @property
    def configuration_id(self) -> str:
        return "__".join(
            (self.regime, self.method, self.controller, self.drive)
        )

    def to_json_dict(self) -> dict[str, str]:
        return {
            "regime": self.regime,
            "drive": self.drive,
            "method": self.method,
            "controller": self.controller,
        }


@dataclass(frozen=True)
class AccuracyMetrics:
    """Post-run exact-reference errors used by the qualification policy."""

    mean_abs_energy_error: float
    mean_abs_doublon_error: float
    mean_max_abs_site_occupation_error: float
    mean_infidelity: float | None

    def __post_init__(self) -> None:
        for name in (
            "mean_abs_energy_error",
            "mean_abs_doublon_error",
            "mean_max_abs_site_occupation_error",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if self.mean_infidelity is not None:
            object.__setattr__(
                self,
                "mean_infidelity",
                _finite_nonnegative(
                    self.mean_infidelity,
                    name="mean_infidelity",
                ),
            )

    def gated_values(self, *, include_infidelity: bool) -> tuple[float, ...]:
        values = (
            self.mean_abs_energy_error,
            self.mean_abs_doublon_error,
            self.mean_max_abs_site_occupation_error,
        )
        if not include_infidelity:
            return values
        if self.mean_infidelity is None:
            raise ValueError(
                "qualification requires statevector-derived infidelity, but the "
                "run artifact does not record it"
            )
        return (*values, self.mean_infidelity)

    def to_json_dict(self) -> dict[str, float | None]:
        return {
            "mean_abs_energy_error": self.mean_abs_energy_error,
            "mean_abs_doublon_error": self.mean_abs_doublon_error,
            "mean_max_abs_site_occupation_error": (
                self.mean_max_abs_site_occupation_error
            ),
            "mean_infidelity": self.mean_infidelity,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "AccuracyMetrics":
        return cls(
            mean_abs_energy_error=float(payload["mean_abs_energy_error"]),
            mean_abs_doublon_error=float(payload["mean_abs_doublon_error"]),
            mean_max_abs_site_occupation_error=float(
                payload["mean_max_abs_site_occupation_error"]
            ),
            mean_infidelity=(
                None
                if payload.get("mean_infidelity") is None
                else float(payload["mean_infidelity"])
            ),
        )


@dataclass(frozen=True)
class AccuracyPolicy:
    """Scientific admission rule applied to exact-reference metrics.

    ``response_normalized_matched_pair`` measures observable errors relative
    to the RMS exact response induced by the declared drive, then normalizes
    every primitive error to its target as a dimensionless ``q`` component. A
    run can meet the five-percent response target on its own, while a completed
    APM/AVQDS pair may also be admitted when both arms stay inside a broader
    physical guard and both their RMS burdens and deadbanded components agree
    within the declared factor.

    The older ``componentwise_ceiling`` and ``mean_infidelity_window`` modes
    remain replayable so preserved policy ledgers never lose their meaning.
    """

    ceiling: float = 1.0e-4
    preferred_floor: float = 1.0e-5
    require_infidelity: bool = True
    mode: str = POLICY_COMPONENTWISE_CEILING
    infidelity_floor: float | None = None
    infidelity_ceiling: float | None = None
    response_relative_target: float | None = None
    response_relative_guard: float | None = None
    matched_accuracy_factor: float | None = None
    matched_component_deadband: float = 0.10
    response_scale_floor: float = 1.0e-12
    absolute_energy_error_ceiling: float | None = None
    absolute_doublon_error_ceiling: float | None = None
    absolute_site_occupation_error_ceiling: float | None = None
    response_scales: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode == POLICY_COMPONENTWISE_CEILING:
            ceiling = _finite_nonnegative(self.ceiling, name="ceiling")
            preferred = _finite_nonnegative(
                self.preferred_floor,
                name="preferred_floor",
            )
            if ceiling <= 0.0:
                raise ValueError("accuracy ceiling must be positive")
            if preferred > ceiling:
                raise ValueError(
                    "preferred_floor cannot exceed the accuracy ceiling"
                )
            return
        if self.mode == POLICY_RESPONSE_NORMALIZED_MATCHED:
            if not self.require_infidelity:
                raise ValueError(
                    "a response-normalized matched policy requires infidelity"
                )
            required = {
                "response_relative_target": self.response_relative_target,
                "response_relative_guard": self.response_relative_guard,
                "matched_accuracy_factor": self.matched_accuracy_factor,
                "infidelity_ceiling": self.infidelity_ceiling,
                "absolute_energy_error_ceiling": (
                    self.absolute_energy_error_ceiling
                ),
                "absolute_doublon_error_ceiling": (
                    self.absolute_doublon_error_ceiling
                ),
                "absolute_site_occupation_error_ceiling": (
                    self.absolute_site_occupation_error_ceiling
                ),
            }
            values = {
                name: _finite_nonnegative(value, name=name)
                for name, value in required.items()
                if value is not None
            }
            if len(values) != len(required) or any(value <= 0.0 for value in values.values()):
                raise ValueError(
                    "a response-normalized matched policy requires positive "
                    "targets, guards, matching factor, infidelity ceiling, and "
                    "absolute safety ceilings"
                )
            assert self.response_relative_target is not None
            assert self.response_relative_guard is not None
            assert self.matched_accuracy_factor is not None
            if self.response_relative_target >= self.response_relative_guard:
                raise ValueError(
                    "response_relative_target must be below the comparison guard"
                )
            if self.matched_accuracy_factor < 1.0:
                raise ValueError("matched_accuracy_factor must be at least one")
            deadband = _finite_nonnegative(
                self.matched_component_deadband,
                name="matched_component_deadband",
            )
            if deadband <= 0.0:
                raise ValueError("matched_component_deadband must be positive")
            floor = _finite_nonnegative(
                self.response_scale_floor,
                name="response_scale_floor",
            )
            if floor <= 0.0:
                raise ValueError("response_scale_floor must be positive")
            if not self.response_scales:
                raise ValueError(
                    "a response-normalized matched policy requires exact-response scales"
                )
            if is_response_scale_index(self.response_scales):
                validate_response_scale_index(self.response_scales)
            else:
                for drive, horizons in self.response_scales.items():
                    if not str(drive).strip() or not isinstance(horizons, Mapping):
                        raise ValueError("response-scale drives must be named mappings")
                    for horizon in ("t5", "t20"):
                        row = horizons.get(horizon)
                        if not isinstance(row, Mapping):
                            raise ValueError(
                                f"response scales for {drive!r} omit {horizon!r}"
                            )
                        for name in (
                            "energy_rms_excursion",
                            "doublon_rms_excursion",
                            "max_site_occupation_rms_excursion",
                        ):
                            if _finite_nonnegative(row.get(name), name=name) <= 0.0:
                                raise ValueError(
                                    f"response scale {drive}/{horizon}/{name} must be positive"
                                )
            return
        if self.mode != POLICY_MEAN_INFIDELITY_WINDOW:
            raise ValueError(f"unsupported accuracy-policy mode: {self.mode!r}")
        if not self.require_infidelity:
            raise ValueError("an infidelity-window policy requires infidelity")
        if self.infidelity_floor is None or self.infidelity_ceiling is None:
            raise ValueError("an infidelity-window policy requires both bounds")
        floor = _finite_nonnegative(
            self.infidelity_floor,
            name="infidelity_floor",
        )
        ceiling = _finite_nonnegative(
            self.infidelity_ceiling,
            name="infidelity_ceiling",
        )
        if floor <= 0.0 or ceiling <= 0.0:
            raise ValueError("infidelity-window bounds must be positive")
        if floor >= ceiling:
            raise ValueError("infidelity_floor must be below infidelity_ceiling")

    @classmethod
    def mean_infidelity_window(
        cls,
        *,
        floor: float,
        ceiling: float,
    ) -> "AccuracyPolicy":
        return cls(
            mode=POLICY_MEAN_INFIDELITY_WINDOW,
            require_infidelity=True,
            infidelity_floor=floor,
            infidelity_ceiling=ceiling,
        )

    @classmethod
    def response_normalized_matched_pair(
        cls,
        *,
        response_scales: Mapping[str, Any],
        response_relative_target: float = 0.05,
        response_relative_guard: float = 0.15,
        matched_accuracy_factor: float = 2.0,
        matched_component_deadband: float = 0.10,
        infidelity_ceiling: float = 1.0e-4,
        absolute_energy_error_ceiling: float = 1.0e-2,
        absolute_doublon_error_ceiling: float = 5.0e-3,
        absolute_site_occupation_error_ceiling: float = 5.0e-3,
        response_scale_floor: float = 1.0e-12,
    ) -> "AccuracyPolicy":
        return cls(
            mode=POLICY_RESPONSE_NORMALIZED_MATCHED,
            require_infidelity=True,
            infidelity_ceiling=infidelity_ceiling,
            response_relative_target=response_relative_target,
            response_relative_guard=response_relative_guard,
            matched_accuracy_factor=matched_accuracy_factor,
            matched_component_deadband=matched_component_deadband,
            response_scale_floor=response_scale_floor,
            absolute_energy_error_ceiling=absolute_energy_error_ceiling,
            absolute_doublon_error_ceiling=absolute_doublon_error_ceiling,
            absolute_site_occupation_error_ceiling=(
                absolute_site_occupation_error_ceiling
            ),
            response_scales=dict(response_scales),
        )

    def _infidelity(self, metrics: AccuracyMetrics) -> float:
        if metrics.mean_infidelity is None:
            raise ValueError(
                "qualification requires statevector-derived infidelity, but the "
                "run artifact does not record it"
            )
        return float(metrics.mean_infidelity)

    def _response_scale_row(
        self,
        *,
        key: CellKey | None,
        horizon: str | None,
    ) -> Mapping[str, Any]:
        if key is None or horizon is None:
            raise ValueError(
                "response-normalized qualification requires a cell key and horizon"
            )
        if is_response_scale_index(self.response_scales):
            return response_scale_row(
                self.response_scales,
                regime=key.regime,
                controller=key.controller,
                drive=key.drive,
                horizon=horizon,
            )
        try:
            return self.response_scales[key.drive][horizon]
        except KeyError as exc:
            raise ValueError(
                f"no exact-response scale is bound for {key.drive}/{horizon}"
            ) from exc

    def response_components(
        self,
        metrics: AccuracyMetrics,
        *,
        key: CellKey,
        horizon: str,
    ) -> dict[str, float]:
        """Return the target-normalized ``q`` vector used for matching.

        The raw absolute errors remain unchanged on :class:`AccuracyMetrics`.
        Observable components first form the exact-response-relative error
        ``r_O`` and then divide by the declared individual response target;
        fidelity divides by its declared ceiling.
        """

        if self.mode != POLICY_RESPONSE_NORMALIZED_MATCHED:
            raise ValueError("response components require the matched-pair policy")
        row = self._response_scale_row(key=key, horizon=horizon)
        floor = float(self.response_scale_floor)
        assert self.response_relative_target is not None
        assert self.infidelity_ceiling is not None
        response_target = float(self.response_relative_target)
        return {
            "mean_abs_energy_error": (
                float(metrics.mean_abs_energy_error)
                / max(float(row["energy_rms_excursion"]), floor)
                / response_target
            ),
            "mean_abs_doublon_error": (
                float(metrics.mean_abs_doublon_error)
                / max(float(row["doublon_rms_excursion"]), floor)
                / response_target
            ),
            "mean_max_abs_site_occupation_error": (
                float(metrics.mean_max_abs_site_occupation_error)
                / max(float(row["max_site_occupation_rms_excursion"]), floor)
                / response_target
            ),
            "mean_infidelity": self._infidelity(metrics)
            / float(self.infidelity_ceiling),
        }

    def _absolute_safety_passes(self, metrics: AccuracyMetrics) -> bool:
        assert self.absolute_energy_error_ceiling is not None
        assert self.absolute_doublon_error_ceiling is not None
        assert self.absolute_site_occupation_error_ceiling is not None
        return (
            metrics.mean_abs_energy_error
            <= float(self.absolute_energy_error_ceiling)
            and metrics.mean_abs_doublon_error
            <= float(self.absolute_doublon_error_ceiling)
            and metrics.mean_max_abs_site_occupation_error
            <= float(self.absolute_site_occupation_error_ceiling)
        )

    def comparison_eligible(
        self,
        metrics: AccuracyMetrics,
        *,
        key: CellKey,
        horizon: str,
    ) -> bool:
        if self.mode != POLICY_RESPONSE_NORMALIZED_MATCHED:
            return self.qualifies(metrics, key=key, horizon=horizon)
        assert self.response_relative_guard is not None
        assert self.response_relative_target is not None
        components = self.response_components(metrics, key=key, horizon=horizon)
        broad_q_ceiling = float(self.response_relative_guard) / float(
            self.response_relative_target
        )
        return self._absolute_safety_passes(metrics) and all(
            value <= broad_q_ceiling
            for value in components.values()
        )

    def accuracy_burden(
        self,
        metrics: AccuracyMetrics,
        *,
        key: CellKey,
        horizon: str,
    ) -> float:
        components = self.response_components(metrics, key=key, horizon=horizon)
        return math.sqrt(
            sum(value * value for value in components.values())
            / float(len(components))
        )

    def pair_matches(
        self,
        left_metrics: AccuracyMetrics,
        *,
        left_key: CellKey,
        right_metrics: AccuracyMetrics,
        right_key: CellKey,
        horizon: str,
    ) -> bool:
        if self.mode != POLICY_RESPONSE_NORMALIZED_MATCHED:
            return False
        if (
            left_key.regime != right_key.regime
            or left_key.drive != right_key.drive
            or left_key.controller != right_key.controller
            or left_key.method == right_key.method
        ):
            return False
        if not self.comparison_eligible(
            left_metrics, key=left_key, horizon=horizon
        ) or not self.comparison_eligible(
            right_metrics, key=right_key, horizon=horizon
        ):
            return False
        left_components = self.response_components(
            left_metrics,
            key=left_key,
            horizon=horizon,
        )
        right_components = self.response_components(
            right_metrics,
            key=right_key,
            horizon=horizon,
        )
        if set(left_components) != set(right_components):
            return False
        left = self.accuracy_burden(
            left_metrics, key=left_key, horizon=horizon
        )
        right = self.accuracy_burden(
            right_metrics, key=right_key, horizon=horizon
        )
        burden_floor = float(self.response_scale_floor)
        left = max(left, burden_floor)
        right = max(right, burden_floor)
        assert self.matched_accuracy_factor is not None
        factor = float(self.matched_accuracy_factor)
        burdens_match = max(left / right, right / left) <= factor
        deadband = float(self.matched_component_deadband)
        components_match = all(
            max(
                max(float(left_components[name]), deadband)
                / max(float(right_components[name]), deadband),
                max(float(right_components[name]), deadband)
                / max(float(left_components[name]), deadband),
            )
            <= factor
            for name in left_components
        )
        return burdens_match and components_match

    def qualifies(
        self,
        metrics: AccuracyMetrics,
        *,
        key: CellKey | None = None,
        horizon: str | None = None,
    ) -> bool:
        if self.mode == POLICY_MEAN_INFIDELITY_WINDOW:
            value = self._infidelity(metrics)
            assert self.infidelity_floor is not None
            assert self.infidelity_ceiling is not None
            return (
                float(self.infidelity_floor)
                <= value
                <= float(self.infidelity_ceiling)
            )
        if self.mode == POLICY_RESPONSE_NORMALIZED_MATCHED:
            assert key is not None
            assert horizon is not None
            assert self.response_relative_target is not None
            components = self.response_components(
                metrics,
                key=key,
                horizon=horizon,
            )
            return self._absolute_safety_passes(metrics) and all(
                value <= 1.0 for value in components.values()
            )
        return all(
            value <= float(self.ceiling)
            for value in metrics.gated_values(
                include_infidelity=bool(self.require_infidelity)
            )
        )

    def classify(
        self,
        metrics: AccuracyMetrics,
        *,
        key: CellKey | None = None,
        horizon: str | None = None,
    ) -> str:
        if self.qualifies(metrics, key=key, horizon=horizon):
            return STATUS_ACCURACY_QUALIFIED
        if self.mode == POLICY_MEAN_INFIDELITY_WINDOW:
            assert self.infidelity_floor is not None
            return (
                STATUS_REQUIRES_THRESHOLD_COARSENING
                if self._infidelity(metrics) < float(self.infidelity_floor)
                else STATUS_REQUIRES_THRESHOLD_EXTENSION
            )
        return STATUS_REQUIRES_THRESHOLD_EXTENSION

    def band(
        self,
        metrics: AccuracyMetrics,
        *,
        key: CellKey | None = None,
        horizon: str | None = None,
    ) -> str:
        """Describe accuracy without changing qualification or promotion."""

        if self.mode == POLICY_MEAN_INFIDELITY_WINDOW:
            value = self._infidelity(metrics)
            assert self.infidelity_floor is not None
            assert self.infidelity_ceiling is not None
            if value < float(self.infidelity_floor):
                return "below_infidelity_window"
            if value > float(self.infidelity_ceiling):
                return "above_infidelity_window"
            return "infidelity_window"

        if self.mode == POLICY_RESPONSE_NORMALIZED_MATCHED:
            assert key is not None
            assert horizon is not None
            if self.qualifies(metrics, key=key, horizon=horizon):
                return "response_target"
            if self.comparison_eligible(metrics, key=key, horizon=horizon):
                return "comparison_guard"
            return "outside_comparison_guard"

        worst = max(
            metrics.gated_values(include_infidelity=bool(self.require_infidelity))
        )
        if worst > float(self.ceiling):
            return "above_ceiling"
        if worst < float(self.preferred_floor):
            return "better_than_preferred_band"
        return "preferred_band"

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "require_infidelity": bool(self.require_infidelity),
        }
        if self.mode == POLICY_MEAN_INFIDELITY_WINDOW:
            payload.update(
                {
                    "infidelity_floor": float(self.infidelity_floor),
                    "infidelity_ceiling": float(self.infidelity_ceiling),
                    "observable_errors": "reported_diagnostics_not_admission_gates",
                }
            )
        elif self.mode == POLICY_RESPONSE_NORMALIZED_MATCHED:
            payload.update(
                {
                    "response_relative_target": float(
                        self.response_relative_target
                    ),
                    "response_relative_guard": float(
                        self.response_relative_guard
                    ),
                    "matched_accuracy_factor": float(
                        self.matched_accuracy_factor
                    ),
                    "matched_component_deadband": float(
                        self.matched_component_deadband
                    ),
                    "matched_policy_version": 2,
                    "component_normalization": "target_normalized_q",
                    "aggregate_burden": "rms_q_components",
                    "comparison_guard_components": (
                        "all_target_normalized_q_including_fidelity"
                    ),
                    "absolute_safety_components": [
                        "mean_abs_energy_error",
                        "mean_abs_doublon_error",
                        "mean_max_abs_site_occupation_error",
                    ],
                    "infidelity_role": (
                        "target_normalized_component_with_broad_guard_not_"
                        "absolute_safety"
                    ),
                    "response_scale_floor": float(self.response_scale_floor),
                    "infidelity_ceiling": float(self.infidelity_ceiling),
                    "absolute_energy_error_ceiling": float(
                        self.absolute_energy_error_ceiling
                    ),
                    "absolute_doublon_error_ceiling": float(
                        self.absolute_doublon_error_ceiling
                    ),
                    "absolute_site_occupation_error_ceiling": float(
                        self.absolute_site_occupation_error_ceiling
                    ),
                    "response_scales": dict(self.response_scales),
                    "pair_admission": (
                        "both arms inside the comparison guard, aggregate q "
                        "burdens matched within the declared factor, and every "
                        "deadbanded q component matched within that factor"
                    ),
                }
            )
        else:
            payload.update(
                {
                    "ceiling": float(self.ceiling),
                    "preferred_floor": float(self.preferred_floor),
                }
            )
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "AccuracyPolicy":
        mode = str(payload.get("mode") or POLICY_COMPONENTWISE_CEILING)
        if mode == POLICY_MEAN_INFIDELITY_WINDOW:
            return cls.mean_infidelity_window(
                floor=float(payload["infidelity_floor"]),
                ceiling=float(payload["infidelity_ceiling"]),
            )
        if mode == POLICY_RESPONSE_NORMALIZED_MATCHED:
            return cls.response_normalized_matched_pair(
                response_scales=dict(payload["response_scales"]),
                response_relative_target=float(
                    payload["response_relative_target"]
                ),
                response_relative_guard=float(
                    payload["response_relative_guard"]
                ),
                matched_accuracy_factor=float(payload["matched_accuracy_factor"]),
                matched_component_deadband=float(
                    payload.get("matched_component_deadband", 0.10)
                ),
                infidelity_ceiling=float(payload["infidelity_ceiling"]),
                absolute_energy_error_ceiling=float(
                    payload["absolute_energy_error_ceiling"]
                ),
                absolute_doublon_error_ceiling=float(
                    payload["absolute_doublon_error_ceiling"]
                ),
                absolute_site_occupation_error_ceiling=float(
                    payload["absolute_site_occupation_error_ceiling"]
                ),
                response_scale_floor=float(payload.get("response_scale_floor", 1.0e-12)),
            )
        return cls(
            mode=mode,
            ceiling=float(payload.get("ceiling", 1.0e-4)),
            preferred_floor=float(payload.get("preferred_floor", 1.0e-5)),
            require_infidelity=bool(payload.get("require_infidelity", True)),
        )


@dataclass(frozen=True)
class RungRecord:
    """One completed threshold rung or one explicit safeguard event."""

    key: CellKey
    horizon: str
    threshold: float
    status: str
    run_path: str | None = None
    metrics: AccuracyMetrics | None = None
    reason: str = ""
    resume_from_run_path: str | None = None
    achieved_configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", _canonical_threshold(self.threshold))
        if self.horizon not in ("t5", "t20"):
            raise ValueError(f"unsupported campaign horizon: {self.horizon!r}")
        if self.status not in RUNG_STATUSES:
            raise ValueError(f"unsupported rung status: {self.status!r}")
        if self.status == STATUS_SAFEGUARD_TRIPPED:
            if self.metrics is not None:
                raise ValueError("a safeguard record cannot carry accuracy metrics")
            if not self.reason.strip():
                raise ValueError("a safeguard record requires a reason")
        elif self.metrics is None or not str(self.run_path or "").strip():
            raise ValueError("a completed rung requires metrics and run_path")
        if self.horizon == "t20" and self.status != STATUS_SAFEGUARD_TRIPPED:
            if not str(self.resume_from_run_path or "").strip():
                raise ValueError("a t20 rung must identify its t5 resume source")

    @classmethod
    def completed(
        cls,
        *,
        key: CellKey,
        horizon: str,
        threshold: float,
        run_path: str,
        metrics: AccuracyMetrics,
        policy: AccuracyPolicy,
        resume_from_run_path: str | None = None,
        achieved_configuration: Mapping[str, Any] | None = None,
    ) -> "RungRecord":
        status = policy.classify(metrics, key=key, horizon=horizon)
        return cls(
            key=key,
            horizon=horizon,
            threshold=threshold,
            status=status,
            run_path=run_path,
            metrics=metrics,
            resume_from_run_path=resume_from_run_path,
            achieved_configuration=dict(achieved_configuration or {}),
            reason=(
                "target_qualified"
                if policy.mode == POLICY_RESPONSE_NORMALIZED_MATCHED
                and status == STATUS_ACCURACY_QUALIFIED
                else ""
            ),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_json_dict(),
            "horizon": self.horizon,
            "threshold": self.threshold,
            "status": self.status,
            "run_path": self.run_path,
            "metrics": None if self.metrics is None else self.metrics.to_json_dict(),
            "reason": self.reason,
            "resume_from_run_path": self.resume_from_run_path,
            "achieved_configuration": dict(self.achieved_configuration),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "RungRecord":
        key = dict(payload["key"])
        metrics = payload.get("metrics")
        return cls(
            key=CellKey(
                regime=str(key["regime"]),
                drive=str(key["drive"]),
                method=str(key["method"]),
                controller=str(key["controller"]),
            ),
            horizon=str(payload["horizon"]),
            threshold=float(payload["threshold"]),
            status=str(payload["status"]),
            run_path=(
                None if payload.get("run_path") is None else str(payload["run_path"])
            ),
            metrics=(
                None if metrics is None else AccuracyMetrics.from_json_dict(metrics)
            ),
            reason=str(payload.get("reason") or ""),
            resume_from_run_path=(
                None
                if payload.get("resume_from_run_path") is None
                else str(payload["resume_from_run_path"])
            ),
            achieved_configuration=dict(payload.get("achieved_configuration") or {}),
        )


@dataclass(frozen=True)
class CampaignLedger:
    """Immutable append-only record of every attempted scientific rung."""

    records: tuple[RungRecord, ...] = ()
    qualification_policy: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "qualification_policy",
            dict(self.qualification_policy),
        )

    def append(self, record: RungRecord) -> "CampaignLedger":
        for existing in self.records:
            if (
                existing.key == record.key
                and existing.horizon == record.horizon
                and _same_threshold(existing.threshold, record.threshold)
            ):
                if existing == record:
                    return self
                raise ValueError(
                    "ledger already contains a different record for "
                    f"{record.key.configuration_id}/{record.horizon}/"
                    f"{record.threshold:.1e}"
                )
        return CampaignLedger(
            (*self.records, record),
            qualification_policy=self.qualification_policy,
        )

    def for_cell(self, key: CellKey, horizon: str) -> tuple[RungRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.key == key and record.horizon == horizon
        )

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "schema": LEDGER_SCHEMA_V1,
            "records": [record.to_json_dict() for record in self.records],
        }
        if self.qualification_policy:
            payload["qualification_policy"] = dict(self.qualification_policy)
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "CampaignLedger":
        if str(payload.get("schema")) != LEDGER_SCHEMA_V1:
            raise ValueError(f"unsupported campaign-ledger schema: {payload.get('schema')!r}")
        ledger = cls(
            qualification_policy=dict(payload.get("qualification_policy") or {})
        )
        for row in payload.get("records") or ():
            ledger = ledger.append(RungRecord.from_json_dict(row))
        return ledger

    def reclassified(self, policy: AccuracyPolicy) -> "CampaignLedger":
        """Replay stored metrics under ``policy`` without touching artifacts."""

        replayed: list[RungRecord] = []
        for record in self.records:
            if record.status == STATUS_SAFEGUARD_TRIPPED:
                replayed.append(record)
                continue
            status = policy.classify(
                record.metrics,
                key=record.key,
                horizon=record.horizon,
            )
            replayed.append(
                replace(
                    record,
                    status=status,
                    reason=(
                        "target_qualified"
                        if policy.mode == POLICY_RESPONSE_NORMALIZED_MATCHED
                        and status == STATUS_ACCURACY_QUALIFIED
                        else ""
                    ),
                )
            )
        records = tuple(replayed)
        if policy.mode == POLICY_RESPONSE_NORMALIZED_MATCHED:
            latest: dict[
                tuple[str, str, str, str, str], tuple[int, RungRecord]
            ] = {}
            for index, record in enumerate(records):
                if record.metrics is None:
                    continue
                identity = (
                    record.key.regime,
                    record.key.drive,
                    record.key.controller,
                    record.horizon,
                    record.key.method,
                )
                latest[identity] = (index, record)
            mutable = list(records)
            physical_cells = {
                (regime, drive, controller, horizon)
                for regime, drive, controller, horizon, _method in latest
            }
            for regime, drive, controller, horizon in physical_cells:
                apm = latest.get(
                    (regime, drive, controller, horizon, METHOD_APM)
                )
                avqds = latest.get(
                    (regime, drive, controller, horizon, METHOD_AVQDS)
                )
                if apm is None or avqds is None:
                    continue
                apm_index, apm_record = apm
                avqds_index, avqds_record = avqds
                assert apm_record.metrics is not None
                assert avqds_record.metrics is not None
                if not policy.pair_matches(
                    apm_record.metrics,
                    left_key=apm_record.key,
                    right_metrics=avqds_record.metrics,
                    right_key=avqds_record.key,
                    horizon=horizon,
                ):
                    continue
                apm_burden = policy.accuracy_burden(
                    apm_record.metrics,
                    key=apm_record.key,
                    horizon=horizon,
                )
                avqds_burden = policy.accuracy_burden(
                    avqds_record.metrics,
                    key=avqds_record.key,
                    horizon=horizon,
                )
                burden_floor = float(policy.response_scale_floor)
                apm_burden = max(apm_burden, burden_floor)
                avqds_burden = max(avqds_burden, burden_floor)
                factor = max(
                    apm_burden / avqds_burden,
                    avqds_burden / apm_burden,
                )
                reason = (
                    "matched_guard: response-normalized APM/AVQDS accuracy "
                    f"burdens matched within factor {factor:.6g}"
                )
                mutable[apm_index] = replace(
                    apm_record,
                    status=STATUS_ACCURACY_QUALIFIED,
                    reason=reason,
                )
                mutable[avqds_index] = replace(
                    avqds_record,
                    status=STATUS_ACCURACY_QUALIFIED,
                    reason=reason,
                )
            records = tuple(mutable)
        return CampaignLedger(
            records,
            qualification_policy=policy.to_json_dict(),
        )

    @classmethod
    def read(cls, path: str | Path) -> "CampaignLedger":
        return cls.from_json_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)


@dataclass(frozen=True)
class ThresholdDecision:
    status: str
    threshold: float
    reason: str


@dataclass(frozen=True)
class ThresholdSchedule:
    """Prior-informed ladder followed by deterministic factor-three extension.

    The schedule never reads achieved error and never assumes that error is
    monotone in the activation cut.  It simply returns the next unattempted
    rung until the accuracy policy classifies one as qualified.
    """

    initial_by_cell: Mapping[CellKey, tuple[float, ...]]
    extension_ratio: float = 3.0
    minimum_threshold: float = 1.0e-12
    maximum_threshold: float = 1.0e-2
    maximum_rungs_per_horizon: int = 24

    def __post_init__(self) -> None:
        if float(self.extension_ratio) <= 1.0:
            raise ValueError("extension_ratio must exceed one")
        _canonical_threshold(self.minimum_threshold)
        _canonical_threshold(self.maximum_threshold)
        if float(self.minimum_threshold) >= float(self.maximum_threshold):
            raise ValueError("minimum_threshold must be below maximum_threshold")
        if int(self.maximum_rungs_per_horizon) < 1:
            raise ValueError("maximum_rungs_per_horizon must be positive")
        for key, values in self.initial_by_cell.items():
            if not isinstance(key, CellKey):
                raise TypeError("threshold schedule keys must be CellKey values")
            if not values:
                raise ValueError(f"{key.configuration_id} has no initial threshold")
            canonical = tuple(_canonical_threshold(value) for value in values)
            if len(canonical) != len(set(canonical)):
                raise ValueError(f"{key.configuration_id} repeats a threshold")

    def next(
        self,
        key: CellKey,
        attempted: Sequence[float],
        *,
        direction: str = "tighten",
    ) -> ThresholdDecision:
        attempted_values = tuple(_canonical_threshold(value) for value in attempted)
        def attempted(candidate: float) -> bool:
            return any(_same_threshold(candidate, value) for value in attempted_values)
        if direction not in {"tighten", "coarsen"}:
            raise ValueError(f"unsupported threshold direction: {direction!r}")
        if len(attempted_values) >= int(self.maximum_rungs_per_horizon):
            anchor = (
                max(attempted_values)
                if direction == "coarsen"
                else min(attempted_values)
            )
            candidate = _canonical_threshold(
                anchor * float(self.extension_ratio)
                if direction == "coarsen"
                else anchor / float(self.extension_ratio)
            )
            while attempted(candidate):
                candidate = _canonical_threshold(
                    candidate * float(self.extension_ratio)
                    if direction == "coarsen"
                    else candidate / float(self.extension_ratio)
                )
            return ThresholdDecision(
                STATUS_SAFEGUARD_TRIPPED,
                candidate,
                f"maximum rung count {self.maximum_rungs_per_horizon} reached",
            )
        if not attempted_values:
            return ThresholdDecision(
                STATUS_REQUIRES_THRESHOLD_EXTENSION,
                _canonical_threshold(self.initial_by_cell[key][0]),
                "next prior-informed threshold rung",
            )
        anchor = attempted_values[-1]
        if direction == "coarsen":
            coarser_initial = sorted(
                _canonical_threshold(value)
                for value in self.initial_by_cell[key]
                if _canonical_threshold(value) > anchor
                and not attempted(_canonical_threshold(value))
            )
            if coarser_initial:
                return ThresholdDecision(
                    STATUS_REQUIRES_THRESHOLD_COARSENING,
                    coarser_initial[0],
                    "next registered coarser threshold rung",
                )
            # Walk across already measured adjacent rungs before extending the
            # registered frontier. This avoids a new near-duplicate such as
            # 9e-7 when 1e-6 already proves the neighboring rung.
            anchor = max(attempted_values)
            candidate = _canonical_threshold(
                anchor * float(self.extension_ratio)
            )
            while attempted(candidate):
                candidate = _canonical_threshold(
                    candidate * float(self.extension_ratio)
                )
            if candidate > float(self.maximum_threshold):
                return ThresholdDecision(
                    STATUS_SAFEGUARD_TRIPPED,
                    candidate,
                    f"next threshold {candidate:.3e} is above safeguard "
                    f"{self.maximum_threshold:.3e}",
                )
            return ThresholdDecision(
                STATUS_REQUIRES_THRESHOLD_COARSENING,
                candidate,
                "factor-three threshold coarsening",
            )
        tighter_initial = sorted(
            (
                _canonical_threshold(value)
                for value in self.initial_by_cell[key]
                if _canonical_threshold(value) < anchor
                and not attempted(_canonical_threshold(value))
            ),
            reverse=True,
        )
        if tighter_initial:
            return ThresholdDecision(
                STATUS_REQUIRES_THRESHOLD_EXTENSION,
                tighter_initial[0],
                "next prior-informed threshold rung",
            )
        anchor = min(attempted_values)
        candidate = _canonical_threshold(anchor / float(self.extension_ratio))
        while attempted(candidate):
            candidate = _canonical_threshold(
                candidate / float(self.extension_ratio)
            )
        if candidate < float(self.minimum_threshold):
            return ThresholdDecision(
                STATUS_SAFEGUARD_TRIPPED,
                candidate,
                f"next threshold {candidate:.3e} is below safeguard "
                f"{self.minimum_threshold:.3e}",
            )
        return ThresholdDecision(
            STATUS_REQUIRES_THRESHOLD_EXTENSION,
            candidate,
            "factor-three threshold extension",
        )


@dataclass(frozen=True)
class CellAction:
    """One deterministic executor action emitted from a campaign ledger."""

    action: str
    stage: str
    key: CellKey
    horizon: str
    threshold: float
    reason: str
    resume_from_run_path: str | None = None

    @property
    def action_id(self) -> str:
        cut = f"{self.threshold:.12g}".replace(".", "p")
        return "__".join(
            (
                self.stage,
                self.key.configuration_id,
                self.horizon,
                f"cut{cut}",
            )
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action": self.action,
            "stage": self.stage,
            "key": self.key.to_json_dict(),
            "horizon": self.horizon,
            "threshold": self.threshold,
            "reason": self.reason,
            "resume_from_run_path": self.resume_from_run_path,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "CellAction":
        key = dict(payload["key"])
        action = cls(
            action=str(payload["action"]),
            stage=str(payload["stage"]),
            key=CellKey(
                regime=str(key["regime"]),
                drive=str(key["drive"]),
                method=str(key["method"]),
                controller=str(key["controller"]),
            ),
            horizon=str(payload["horizon"]),
            threshold=float(payload["threshold"]),
            reason=str(payload.get("reason") or ""),
            resume_from_run_path=(
                None
                if payload.get("resume_from_run_path") is None
                else str(payload["resume_from_run_path"])
            ),
        )
        declared_id = payload.get("action_id")
        if declared_id is not None and str(declared_id) != action.action_id:
            raise ValueError("staged action id does not match its scientific fields")
        return action


def _continuation_point_count(*, screen_horizon: str, promotion_horizon: str) -> int:
    screen = _runs.HORIZONS[screen_horizon]
    promotion = _runs.HORIZONS[promotion_horizon]
    if screen.num_times < 2 or promotion.num_times < 2:
        raise ValueError("continuation horizons require at least two points")
    screen_dt = screen.t_final / float(screen.num_times - 1)
    promotion_dt = promotion.t_final / float(promotion.num_times - 1)
    if not math.isclose(screen_dt, promotion_dt, rel_tol=0.0, abs_tol=1.0e-14):
        raise ValueError("screen and promotion horizons do not share one global grid")
    remaining_intervals = round(
        (promotion.t_final - screen.t_final) / promotion_dt
    )
    return int(remaining_intervals + 1)


@dataclass(frozen=True)
class StagedCampaign:
    """Per-cell t5 screening followed by accuracy-gated t20 continuation.

    APM and AVQDS share one screening frontier, but promotion is per cell.  As
    soon as one cell is accuracy-qualified at t5, its decision-identical t20
    continuation becomes available while unrelated cells continue screening.
    This keeps a slow or repeatedly extended cell from withholding already
    qualified evidence.
    """

    cells: tuple[CellKey, ...]
    accuracy_policy: AccuracyPolicy
    threshold_schedule: ThresholdSchedule
    screen_horizon: str = "t5"
    promotion_horizon: str = "t20"
    method_order: tuple[str, ...] = (METHOD_AVQDS, METHOD_APM)
    comparator_first: bool = False
    # The ledger vocabulary keeps one APM method identity ("exchange"); the
    # registered arm that realizes it may be a closed selector bundle such as
    # ``exchange_q_compiled`` (the manuscript's realized Gamma(v*) route on the
    # dual-selector lineage).  The arm id is recorded in every run id and argv.
    apm_arm: str = METHOD_APM

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("staged campaign requires at least one cell")
        if len(self.cells) != len(set(self.cells)):
            raise ValueError("staged campaign repeats a cell")
        if set(self.cells) != set(self.threshold_schedule.initial_by_cell):
            raise ValueError("threshold schedule must cover the campaign exactly")
        if self.screen_horizon not in _runs.HORIZONS:
            raise KeyError(f"unknown screen horizon: {self.screen_horizon!r}")
        if self.promotion_horizon not in _runs.HORIZONS:
            raise KeyError(f"unknown promotion horizon: {self.promotion_horizon!r}")
        if set(self.method_order) != {METHOD_AVQDS, METHOD_APM}:
            raise ValueError("Paper-II staged campaign requires AVQDS and APM")
        apm_arm = _runs.ARMS.get(str(self.apm_arm))
        if apm_arm is None or apm_arm.is_comparator:
            raise ValueError(
                f"apm_arm must name a registered non-comparator arm; got {self.apm_arm!r}"
            )
        _continuation_point_count(
            screen_horizon=self.screen_horizon,
            promotion_horizon=self.promotion_horizon,
        )

    def _method_cells(self, method: str) -> tuple[CellKey, ...]:
        return tuple(sorted(key for key in self.cells if key.method == method))

    @staticmethod
    def _avqds_comparator(key: CellKey) -> CellKey:
        return CellKey(
            regime=key.regime,
            drive=key.drive,
            method=METHOD_AVQDS,
            controller=key.controller,
        )

    @staticmethod
    def _method_peer(key: CellKey) -> CellKey:
        return CellKey(
            regime=key.regime,
            drive=key.drive,
            method=(METHOD_APM if key.method == METHOD_AVQDS else METHOD_AVQDS),
            controller=key.controller,
        )

    @staticmethod
    def _latest_valid(records: Sequence[RungRecord]) -> RungRecord | None:
        return next(
            (record for record in reversed(records) if record.metrics is not None),
            None,
        )

    def _avqds_alignment_direction(
        self,
        *,
        avqds: RungRecord,
        apm: RungRecord,
        horizon: str,
    ) -> str:
        """Choose one adjacent AVQDS rung relative to the frozen APM anchor."""

        assert avqds.metrics is not None
        assert apm.metrics is not None
        policy = self.accuracy_policy
        if not policy.comparison_eligible(
            avqds.metrics, key=avqds.key, horizon=horizon
        ):
            return "tighten"
        avqds_components = policy.response_components(
            avqds.metrics, key=avqds.key, horizon=horizon
        )
        apm_components = policy.response_components(
            apm.metrics, key=apm.key, horizon=horizon
        )
        avqds_burden = policy.accuracy_burden(
            avqds.metrics, key=avqds.key, horizon=horizon
        )
        apm_burden = policy.accuracy_burden(
            apm.metrics, key=apm.key, horizon=horizon
        )
        assert policy.matched_accuracy_factor is not None
        factor = float(policy.matched_accuracy_factor)
        floor = max(float(policy.response_scale_floor), 1.0e-300)
        avqds_burden = max(avqds_burden, floor)
        apm_burden = max(apm_burden, floor)
        if avqds_burden / apm_burden > factor:
            return "tighten"
        if apm_burden / avqds_burden > factor:
            return "coarsen"

        deadband = float(policy.matched_component_deadband)
        mismatches: list[tuple[float, str]] = []
        for name in sorted(avqds_components):
            scout = max(float(avqds_components[name]), deadband)
            anchor = max(float(apm_components[name]), deadband)
            ratio = max(scout / anchor, anchor / scout)
            if ratio > factor:
                mismatches.append((ratio, "tighten" if scout > anchor else "coarsen"))
        if mismatches:
            # The largest primitive mismatch decides the next adjacent rung.
            # The achieved result is always measured; this is not a monotonicity
            # inference or permission to skip intermediate cuts.
            return max(mismatches, key=lambda row: row[0])[1]
        return "tighten" if avqds_burden > apm_burden else "coarsen"

    @staticmethod
    def _qualified(records: Sequence[RungRecord]) -> RungRecord | None:
        return next(
            (
                record
                for record in records
                if record.status == STATUS_ACCURACY_QUALIFIED
            ),
            None,
        )

    @staticmethod
    def _terminal(records: Sequence[RungRecord]) -> bool:
        return any(
            record.status
            in (STATUS_ACCURACY_QUALIFIED, STATUS_SAFEGUARD_TRIPPED)
            for record in records
        )

    def _phase_actions(
        self,
        *,
        ledger: CampaignLedger,
        method: str,
        horizon: str,
        stage: str,
    ) -> tuple[CellAction, ...]:
        actions: list[CellAction] = []
        for key in self._method_cells(method):
            screen_records = ledger.for_cell(key, self.screen_horizon)
            screen_qualified = self._qualified(screen_records)
            if horizon == self.screen_horizon and self._qualified(
                ledger.for_cell(key, self.promotion_horizon)
            ) is not None:
                continue
            if horizon == self.promotion_horizon and screen_qualified is None:
                continue
            if (
                horizon == self.promotion_horizon
                and self.comparator_first
                and self.accuracy_policy.mode == POLICY_RESPONSE_NORMALIZED_MATCHED
            ):
                peer_screen = self._latest_valid(
                    ledger.for_cell(self._method_peer(key), self.screen_horizon)
                )
                if (
                    screen_qualified.metrics is None
                    or peer_screen is None
                    or peer_screen.metrics is None
                    or not self.accuracy_policy.pair_matches(
                        screen_qualified.metrics,
                        left_key=screen_qualified.key,
                        right_metrics=peer_screen.metrics,
                        right_key=peer_screen.key,
                        horizon=self.screen_horizon,
                    )
                ):
                    continue
            records = ledger.for_cell(key, horizon)
            valid_records = tuple(
                record for record in records if record.metrics is not None
            )
            latest = self._latest_valid(records)
            comparator_seed: RungRecord | None = None
            comparator_exhausted = False
            alignment_direction: str | None = None
            if (
                self.comparator_first
                and self.accuracy_policy.mode == POLICY_RESPONSE_NORMALIZED_MATCHED
            ):
                peer = self._method_peer(key)
                peer_records = ledger.for_cell(peer, horizon)
                peer_latest = self._latest_valid(peer_records)
                peer_exhausted = any(
                    record.status == STATUS_SAFEGUARD_TRIPPED
                    for record in peer_records
                )
                pair_matched = (
                    latest is not None
                    and peer_latest is not None
                    and latest.metrics is not None
                    and peer_latest.metrics is not None
                    and self.accuracy_policy.pair_matches(
                        latest.metrics,
                        left_key=latest.key,
                        right_metrics=peer_latest.metrics,
                        right_key=peer_latest.key,
                        horizon=horizon,
                    )
                )
                if pair_matched:
                    continue
                if method == METHOD_APM:
                    comparator_seed = peer_latest
                    comparator_exhausted = peer_exhausted
                    if latest is not None:
                        # APM is the final anchor. Freeze its first admissible
                        # rung while AVQDS moves in either direction. If AVQDS
                        # exhausted outside the guard, retain exactly one APM
                        # diagnostic and make no matched-cost claim.
                        if comparator_exhausted or self.accuracy_policy.comparison_eligible(
                            latest.metrics, key=latest.key, horizon=horizon
                        ):
                            continue
                    elif comparator_seed is None:
                        continue
                    elif not comparator_exhausted and not self.accuracy_policy.comparison_eligible(
                        comparator_seed.metrics,
                        key=comparator_seed.key,
                        horizon=horizon,
                    ):
                        continue
                else:
                    apm_anchor = peer_latest
                    if apm_anchor is not None and self.accuracy_policy.comparison_eligible(
                        apm_anchor.metrics, key=apm_anchor.key, horizon=horizon
                    ):
                        if latest is not None:
                            alignment_direction = self._avqds_alignment_direction(
                                avqds=latest,
                                apm=apm_anchor,
                                horizon=horizon,
                            )
                    elif latest is not None and self.accuracy_policy.comparison_eligible(
                        latest.metrics, key=latest.key, horizon=horizon
                    ):
                        # Scout reached the broad guard; pause it while the APM
                        # anchor is created or calibrated.
                        continue
            elif self._terminal(records):
                continue
            if any(
                record.status == STATUS_SAFEGUARD_TRIPPED for record in records
            ):
                continue
            completed = tuple(
                record
                for record in records
                if record.status != STATUS_SAFEGUARD_TRIPPED
            )
            direction = (
                alignment_direction
                or (
                    "coarsen"
                    if completed
                    and completed[-1].status == STATUS_REQUIRES_THRESHOLD_COARSENING
                    else "tighten"
                )
            )
            decision = self.threshold_schedule.next(
                key,
                [record.threshold for record in records],
                direction=direction,
            )
            if (
                method == METHOD_APM
                and horizon == self.screen_horizon
                and not records
                and comparator_seed is not None
            ):
                decision = ThresholdDecision(
                    STATUS_REQUIRES_THRESHOLD_EXTENSION,
                    comparator_seed.threshold,
                    (
                        "tightest-feasible APM diagnostic after AVQDS ladder "
                        "exhaustion"
                        if comparator_exhausted
                        else "seed APM from the AVQDS broad-guard scout cut"
                    ),
                )
            if horizon == self.promotion_horizon and not records:
                assert screen_qualified is not None
                decision = ThresholdDecision(
                    STATUS_REQUIRES_THRESHOLD_EXTENSION,
                    screen_qualified.threshold,
                    "resume the t5 accuracy-qualified rung",
                )
            action = (
                ACTION_RECORD_SAFEGUARD
                if decision.status == STATUS_SAFEGUARD_TRIPPED
                else ACTION_RUN
            )
            actions.append(
                CellAction(
                    action=action,
                    stage=stage,
                    key=key,
                    horizon=horizon,
                    threshold=decision.threshold,
                    reason=decision.reason,
                    resume_from_run_path=(
                        None
                        if horizon == self.screen_horizon
                        else str(screen_qualified.run_path)
                    ),
                )
            )
        return tuple(actions)

    def next_actions(self, ledger: CampaignLedger) -> tuple[CellAction, ...]:
        """Return every current t5 and t20 action frontier.

        The frontier is crossed over both methods.  Individual action stages
        remain method-specific so previously emitted action receipts retain
        their stable identifiers.
        """

        effective_ledger = (
            ledger.reclassified(self.accuracy_policy)
            if self.accuracy_policy.mode == POLICY_RESPONSE_NORMALIZED_MATCHED
            else ledger
        )
        screening = tuple(
            action
            for method, stage in (
                (METHOD_AVQDS, STAGE_SCREEN_AVQDS),
                (METHOD_APM, STAGE_SCREEN_APM),
            )
            for action in self._phase_actions(
                ledger=effective_ledger,
                method=method,
                horizon=self.screen_horizon,
                stage=stage,
            )
        )
        promotions = tuple(
            action
            for method, stage in (
                (METHOD_AVQDS, STAGE_PROMOTE_AVQDS),
                (METHOD_APM, STAGE_PROMOTE_APM),
            )
            for action in self._phase_actions(
                ledger=effective_ledger,
                method=method,
                horizon=self.promotion_horizon,
                stage=stage,
            )
        )
        return (*screening, *promotions)

    def stage(self, ledger: CampaignLedger) -> str:
        actions = self.next_actions(ledger)
        if not actions:
            return STAGE_COMPLETE
        horizons = {action.horizon for action in actions}
        if horizons == {self.screen_horizon}:
            return STAGE_SCREEN_T5_FRONTIER
        if horizons == {self.promotion_horizon}:
            return STAGE_PROMOTE_T20_FRONTIER
        if horizons == {self.screen_horizon, self.promotion_horizon}:
            return STAGE_ACTIVE_FRONTIER
        raise RuntimeError("staged campaign emitted an unknown horizon frontier")

    def build_run_command(
        self,
        action: CellAction,
        *,
        output_json: str,
        numerics: str,
        gate: str,
        search_profile: str,
        seed_path: str | None = None,
    ) -> _runs.RunCommand:
        """Resolve an action through the authoritative Paper-II run registry."""

        if action.action != ACTION_RUN:
            raise ValueError("a safeguard action has no runner command")
        extra_flags: list[str] = ["--record-statevector"]
        if action.resume_from_run_path is not None:
            extra_flags.extend(
                (
                    "--resume-from-run-json",
                    str(action.resume_from_run_path),
                    "--num-times",
                    str(
                        _continuation_point_count(
                            screen_horizon=self.screen_horizon,
                            promotion_horizon=self.promotion_horizon,
                        )
                    ),
                )
            )
        kwargs: dict[str, Any] = {
            "arm": (
                str(self.apm_arm)
                if action.key.method == METHOD_APM
                else action.key.method
            ),
            "drive": action.key.drive,
            "horizon": action.horizon,
            "output_json": output_json,
            "gate": gate,
            "numerics": numerics,
            "step_control": action.key.controller,
            "search_profile": search_profile,
            "activation_cut": action.threshold,
            "extra_flags": tuple(extra_flags),
        }
        if seed_path is None:
            kwargs["regime"] = action.key.regime
        else:
            kwargs["seed_path"] = seed_path
        return _runs.build_run(**kwargs)


def factorial_cells(
    *,
    regime: str,
    drives: Sequence[str],
    controllers: Sequence[str],
    methods: Sequence[str] = (METHOD_AVQDS, METHOD_APM),
) -> tuple[CellKey, ...]:
    """Construct the declared method x controller x drive matrix."""

    return tuple(
        CellKey(regime, drive, method, controller)
        for method in methods
        for controller in controllers
        for drive in drives
    )


__all__ = [
    "ACTION_RECORD_SAFEGUARD",
    "ACTION_RUN",
    "AccuracyMetrics",
    "AccuracyPolicy",
    "CampaignLedger",
    "CellAction",
    "CellKey",
    "LEDGER_SCHEMA_V1",
    "METHOD_APM",
    "METHOD_AVQDS",
    "POLICY_COMPONENTWISE_CEILING",
    "POLICY_MEAN_INFIDELITY_WINDOW",
    "RungRecord",
    "STAGE_ACTIVE_FRONTIER",
    "STAGE_COMPLETE",
    "STAGE_PROMOTE_APM",
    "STAGE_PROMOTE_AVQDS",
    "STAGE_SCREEN_APM",
    "STAGE_SCREEN_AVQDS",
    "STATUS_ACCURACY_QUALIFIED",
    "STATUS_REQUIRES_THRESHOLD_COARSENING",
    "STATUS_REQUIRES_THRESHOLD_EXTENSION",
    "STATUS_SAFEGUARD_TRIPPED",
    "StagedCampaign",
    "ThresholdDecision",
    "ThresholdSchedule",
    "factorial_cells",
]
