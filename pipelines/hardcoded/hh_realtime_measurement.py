#!/usr/bin/env python3
"""Exact same-checkpoint reuse and planning helpers for realtime control."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from pipelines.exact_bench.noise_oracle_runtime import (
    OracleConfig,
    _LOCAL_READOUT_STRATEGIES,
    normalize_mitigation_config,
    normalize_symmetry_mitigation_config,
)
from pipelines.hardcoded.hh_continuation_scoring import (
    MeasurementCacheAudit,
    measurement_group_keys_for_term,
)
from pipelines.hardcoded.hh_continuation_types import MeasurementCacheStats
from pipelines.hardcoded.hh_realtime_checkpoint_types import (
    GeometryValueKey,
    MeasurementTierConfig,
    OracleValueKey,
)


class ExactCheckpointValueCache:
    """Exact same-checkpoint cache keyed by controller geometry identity."""

    def __init__(self, *, checkpoint_id: str, grouping_mode: str) -> None:
        self._checkpoint_id = str(checkpoint_id)
        self._grouping_mode = str(grouping_mode)
        self._entries: dict[GeometryValueKey, dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._extensions = 0

    def get_or_compute(
        self,
        key: GeometryValueKey,
        *,
        tier_name: str,
        compute: Callable[[], Any],
    ) -> tuple[Any, bool]:
        if str(key.checkpoint_id) != self._checkpoint_id:
            raise ValueError(
                f"ExactCheckpointValueCache key checkpoint mismatch: {key.checkpoint_id} != {self._checkpoint_id}."
            )
        if str(key.grouping_mode) != self._grouping_mode:
            raise ValueError(
                f"ExactCheckpointValueCache grouping_mode mismatch: {key.grouping_mode} != {self._grouping_mode}."
            )
        record = self._entries.get(key)
        if record is not None:
            self._hits += 1
            if str(tier_name) not in record["tiers"]:
                record["tiers"].add(str(tier_name))
                self._extensions += 1
            return record["value"], True
        self._misses += 1
        value = compute()
        self._entries[key] = {"value": value, "tiers": {str(tier_name)}}
        self._stores += 1
        return value, False

    def summary(self) -> dict[str, Any]:
        total = int(self._hits + self._misses)
        return {
            "checkpoint_id": str(self._checkpoint_id),
            "grouping_mode": str(self._grouping_mode),
            "entries": int(len(self._entries)),
            "hits": int(self._hits),
            "misses": int(self._misses),
            "stores": int(self._stores),
            "extensions": int(self._extensions),
            "hit_rate": (float(self._hits) / float(total) if total > 0 else 0.0),
        }


class OracleCheckpointValueCache:
    """Tier-aware same-checkpoint cache for oracle-side controller evaluations."""

    def __init__(self, *, checkpoint_id: str) -> None:
        self._checkpoint_id = str(checkpoint_id)
        self._entries: dict[OracleValueKey, Any] = {}
        self._hits = 0
        self._misses = 0
        self._stores = 0

    def get_or_compute(
        self,
        key: OracleValueKey,
        *,
        compute: Callable[[], Any],
    ) -> tuple[Any, bool]:
        if str(key.checkpoint_id) != self._checkpoint_id:
            raise ValueError(
                f"OracleCheckpointValueCache key checkpoint mismatch: {key.checkpoint_id} != {self._checkpoint_id}."
            )
        if key in self._entries:
            self._hits += 1
            return self._entries[key], True
        self._misses += 1
        value = compute()
        self._entries[key] = value
        self._stores += 1
        return value, False

    def summary(self) -> dict[str, Any]:
        total = int(self._hits + self._misses)
        return {
            "checkpoint_id": str(self._checkpoint_id),
            "entries": int(len(self._entries)),
            "hits": int(self._hits),
            "misses": int(self._misses),
            "stores": int(self._stores),
            "hit_rate": (float(self._hits) / float(total) if total > 0 else 0.0),
        }


def validate_controller_tiers_mean_only(tiers: tuple[MeasurementTierConfig, ...]) -> None:
    for tier in tiers:
        aggregate = tier.oracle_aggregate
        if aggregate is not None and str(aggregate).strip().lower() != "mean":
            raise ValueError(
                f"Realtime checkpoint controller currently supports only mean aggregate tiers; got {aggregate!r} for {tier.tier_name}."
            )


"""
Oracle base config validation: mean aggregate; local/noisy modes only for controller v1.
"""
def validate_controller_oracle_base_config(base_config: OracleConfig) -> None:
    aggregate = str(base_config.oracle_aggregate).strip().lower()
    if aggregate != "mean":
        raise ValueError(
            f"checkpoint controller oracle_v1 requires oracle_aggregate='mean'; got {base_config.oracle_aggregate!r}."
        )
    noise_mode = str(base_config.noise_mode).strip().lower()
    if noise_mode not in {"ideal", "shots", "aer_noise", "runtime", "backend_scheduled"}:
        raise ValueError(
            f"checkpoint controller oracle_v1 unsupported noise_mode {base_config.noise_mode!r}; expected one of ['aer_noise', 'backend_scheduled', 'ideal', 'runtime', 'shots']."
        )
    mitigation_cfg = normalize_mitigation_config(getattr(base_config, "mitigation", "none"))
    symmetry_cfg = normalize_symmetry_mitigation_config(
        getattr(base_config, "symmetry_mitigation", "off")
    )
    if noise_mode == "backend_scheduled":
        if not bool(base_config.use_fake_backend):
            raise ValueError(
                "checkpoint controller oracle_v1 requires use_fake_backend=True for backend_scheduled mode."
            )
        mitigation_mode = str(mitigation_cfg.get("mode", "none"))
        if mitigation_mode not in {"none", "readout"}:
            raise ValueError(
                "checkpoint controller oracle_v1 backend_scheduled mode supports only mitigation modes 'none' or 'readout'."
            )
        if mitigation_mode == "readout":
            if str(symmetry_cfg.get("mode", "off")) not in {"off", "verify_only"}:
                raise ValueError(
                    "checkpoint controller oracle_v1 backend_scheduled readout mitigation is not combinable with active symmetry mitigation."
                )
            strategy = str(mitigation_cfg.get("local_readout_strategy") or "mthree")
            if strategy not in _LOCAL_READOUT_STRATEGIES:
                raise ValueError(
                    f"checkpoint controller oracle_v1 unsupported backend_scheduled readout strategy {strategy!r}; expected one of {sorted(_LOCAL_READOUT_STRATEGIES)}."
                )


"""
Tier materialization: clone base OracleConfig per controller tier; keep policy above ExpectationOracle.
"""
def build_controller_oracle_tier_configs(
    base_config: OracleConfig,
    tiers: tuple[MeasurementTierConfig, ...],
) -> dict[str, OracleConfig]:
    validate_controller_tiers_mean_only(tiers)
    validate_controller_oracle_base_config(base_config)
    base_shots = max(1, int(base_config.shots))
    base_repeats = max(1, int(base_config.oracle_repeats))
    out: dict[str, OracleConfig] = {}
    confirm_shots_default = max(1024, int(round(float(base_shots) * 0.50)))
    for tier in tiers:
        tier_name = str(tier.tier_name)
        if tier_name == "scout":
            default_shots = max(256, int(round(float(base_shots) * 0.25)))
            default_repeats = 1
        elif tier_name == "confirm":
            default_shots = confirm_shots_default
            default_repeats = max(1, base_repeats)
        elif tier_name == "commit":
            default_shots = max(base_shots, confirm_shots_default)
            default_repeats = max(2, base_repeats)
        else:
            default_shots = base_shots
            default_repeats = base_repeats
        out[tier_name] = replace(
            base_config,
            shots=(int(tier.oracle_shots) if tier.oracle_shots is not None else int(default_shots)),
            oracle_repeats=(
                int(tier.oracle_repeats)
                if tier.oracle_repeats is not None
                else int(default_repeats)
            ),
            oracle_aggregate=(
                str(tier.oracle_aggregate)
                if tier.oracle_aggregate is not None
                else "mean"
            ),
        )
    return out


def planning_group_keys_for_term(term: Any) -> list[str]:
    return list(measurement_group_keys_for_term(term))


def planning_stats_for_term(term: Any, audit: MeasurementCacheAudit) -> MeasurementCacheStats:
    return audit.estimate(planning_group_keys_for_term(term))


__all__ = [
    "ExactCheckpointValueCache",
    "OracleCheckpointValueCache",
    "build_controller_oracle_tier_configs",
    "planning_group_keys_for_term",
    "planning_stats_for_term",
    "validate_controller_oracle_base_config",
    "validate_controller_tiers_mean_only",
]
