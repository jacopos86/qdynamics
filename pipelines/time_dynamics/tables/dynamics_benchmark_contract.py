#!/usr/bin/env python3
"""Stable JSON contracts for generic dynamics benchmark rows/tables.

This module is intentionally data-only.  It defines the row and table field
shape consumed by manifest/reporting layers without running physics kernels or
changing realtime controller behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

DYNAMICS_BENCHMARK_ROW_SCHEMA = "dynamics_benchmark_row_v1"
DYNAMICS_TABLE_BUNDLE_SCHEMA = "dynamics_table_bundle_v1"
DYNAMICS_TUNING_PROVENANCE_SCHEMA = "dynamics_tuning_provenance_v2"
DYNAMICS_TUNING_GRANULARITY_CLASS = "coarse_hamiltonian_class"
DYNAMICS_TUNING_CLASS_SOURCE = "coarse_hamiltonian_class_policy_v1"
DYNAMICS_STATIC_SCAFFOLD_SCOPE_BENCHMARK_POINT = "benchmark_point"
DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE = "paper_ii_class_tuning_defaults_v1"
DYNAMICS_CASE_METADATA_OVERRIDE_SOURCE = "case_metadata_override_provisional"
DYNAMICS_LEGACY_MISSING_TUNING_SOURCE = "legacy_missing_tuning_provenance"
DYNAMICS_SKIPPED_TUNING_SOURCE = "not_run_skipped"
DYNAMICS_HH_LEGACY_TUNING_SOURCE = "hh_legacy_not_class_locked"
DYNAMICS_SETTINGS_KIND_CONTROLLER = "controller"
DYNAMICS_SETTINGS_KIND_MCLACHLAN = "mclachlan"
DYNAMICS_SETTINGS_KIND_REFERENCE = "reference"
DYNAMICS_SETTINGS_KIND_COMPARATOR = "comparator"
DYNAMICS_SETTINGS_KIND_BENCHMARK = "benchmark"
DYNAMICS_CLASS_SETTINGS_KINDS: tuple[str, ...] = (
    DYNAMICS_SETTINGS_KIND_CONTROLLER,
    DYNAMICS_SETTINGS_KIND_MCLACHLAN,
    DYNAMICS_SETTINGS_KIND_REFERENCE,
    DYNAMICS_SETTINGS_KIND_COMPARATOR,
    DYNAMICS_SETTINGS_KIND_BENCHMARK,
)
DYNAMICS_CANONICAL_CONTROLLER_ALGORITHM_ID = "dyn_controller_full"
DYNAMICS_CANONICAL_CONTROLLER_VARIANT_ID = "full_controller"
DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND: dict[str, str] = {
    "dyn_controller_full": DYNAMICS_SETTINGS_KIND_CONTROLLER,
    "dyn_fixed_mclachlan": DYNAMICS_SETTINGS_KIND_MCLACHLAN,
    "dyn_product_formula_envelope": DYNAMICS_SETTINGS_KIND_COMPARATOR,
    "dyn_qdrift": DYNAMICS_SETTINGS_KIND_COMPARATOR,
    "dyn_fixed_pvqd": DYNAMICS_SETTINGS_KIND_COMPARATOR,
    "dyn_adaptive_pvqd": DYNAMICS_SETTINGS_KIND_COMPARATOR,
    "dyn_avqds": DYNAMICS_SETTINGS_KIND_COMPARATOR,
    "dyn_avqds_tetris": DYNAMICS_SETTINGS_KIND_COMPARATOR,
}
DYNAMICS_TABLE_I_ALGORITHMS: tuple[str, ...] = tuple(DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND)
QISKIT_COMMUNITY_DYNAMICS_ALGORITHM_SETTINGS_KIND: dict[str, str] = {
    "dyn_qiskit_trotter_qrte": DYNAMICS_SETTINGS_KIND_COMPARATOR,
    "dyn_qiskit_pvqd": DYNAMICS_SETTINGS_KIND_COMPARATOR,
    "dyn_qiskit_varqrte": DYNAMICS_SETTINGS_KIND_COMPARATOR,
}
QISKIT_COMMUNITY_DYNAMICS_ALGORITHMS: tuple[str, ...] = tuple(
    QISKIT_COMMUNITY_DYNAMICS_ALGORITHM_SETTINGS_KIND
)
DYNAMICS_ALL_ALGORITHM_CLASS_SETTINGS_CANDIDATE_SOURCE = (
    "paper_ii_all_algorithm_class_calibration_candidate_v1"
)
DYNAMICS_COARSE_TUNING_CLASSES: tuple[str, ...] = (
    "fermionic",
    "bosonic",
    "hybrid",
)
DYNAMICS_TUNING_CLASS_ALIASES: dict[str, str] = {
    "mixed": "hybrid",
    "mixed_fermion_boson": "hybrid",
    "fermion_boson": "hybrid",
}
DYNAMICS_COARSE_TUNING_CLASS_BY_FAMILY: dict[str, str] = {
    "hubbard": "fermionic",
    "ionic_hubbard": "fermionic",
    "extended_hubbard": "fermionic",
    "ttprime_hubbard": "fermionic",
    "spinless_tv": "fermionic",
    "molecular_restricted_closed_shell": "fermionic",
    "spin_boson": "hybrid",
    "bose_hubbard": "bosonic",
    "harmonic_kerr_chain": "bosonic",
    "hh": "hybrid",
    "vibronic_h2": "hybrid",
    "molecular_vibronic_h2": "hybrid",
}

_FORBIDDEN_TUNING_ID_KEYS = {
    "artifact_json",
    "artifact_path",
    "case_id",
    "command",
    "drive_A",
    "drive_custom_weights",
    "drive_omega",
    "drive_pattern",
    "drive_phi",
    "drive_t0",
    "drive_tbar",
    "enable_drive",
    "family",
    "family_key",
    "family_instance",
    "hamiltonian_coefficients",
    "num_times",
    "output_dir",
    "problem_family",
    "record_id",
    "run_tag",
    "same_seed_comparator_group_id",
    "seed",
    "seed_artifact_sha256",
    "source_artifact_json",
    "source_table_class",
    "static_seed_artifact_json",
    "static_seed_artifact_sha256",
    "table_class",
    "t_final",
    "time_grid",
}
DYNAMICS_FORBIDDEN_TUNING_ID_KEYS = frozenset(_FORBIDDEN_TUNING_ID_KEYS)

DYNAMICS_TABLE_FIELD_KEYS: tuple[str, ...] = (
    "mean_abs_energy_total_error",
    "epsilon_obs_2",
    "one_minus_min_fidelity_exact",
    "epsilon_spec",
    "compiled_count_2q_total",
    "compiled_depth_2q_total",
    "compiled_depth_total",
    "shots_total",
    "table_status_label",
)

DYNAMICS_ROW_STATUS_VALUES: tuple[str, ...] = (
    "completed",
    "skipped_unsupported",
    "skipped_not_implemented",
    "skipped_no_runner",
    "failed",
)

_METRIC_TABLE_MIRROR_KEYS: tuple[str, ...] = (
    "mean_abs_energy_total_error",
    "epsilon_obs_2",
    "one_minus_min_fidelity_exact",
)
_RESOURCE_TABLE_MIRROR_KEYS: tuple[str, ...] = (
    "compiled_count_2q_total",
    "compiled_depth_2q_total",
    "compiled_depth_total",
    "shots_total",
)

TABLE_IV_PRUNE_PILOT_AGGREGATION_SCOPE = "hubbard_A0p2_single_pair_pilot"
TABLE_IV_PRUNE_PILOT_STATUS = "paired_pilot"

_PRUNE_ENABLED_PILOT = {
    "mean_abs_energy_total_error": 7.85982671584634e-06,
    "primary_observable_mae_span": 0.004244803721153237,
    "prune_count": 9,
    "final_params": 32,
}
_NO_PRUNE_PILOT = {
    "mean_abs_energy_total_error": 7.860575173008284e-06,
    "primary_observable_mae_span": 0.004244505386009453,
    "prune_count": 0,
    "final_params": 56,
}


def json_safe(value: Any) -> Any:
    """Return a recursively JSON-safe value, replacing NaN/inf with ``None``."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, complex):
        return {"re": json_safe(float(value.real)), "im": json_safe(float(value.imag))}
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:  # pragma: no cover - defensive for foreign scalar types
            return str(value)
    return value


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return float(out) if math.isfinite(out) else None


def _values_match(left: Any, right: Any, *, atol: float = 1e-12) -> bool:
    if left is None and right is None:
        return True
    lf = _finite_float_or_none(left)
    rf = _finite_float_or_none(right)
    if lf is not None and rf is not None:
        return abs(lf - rf) <= float(atol)
    return left == right


def validate_dynamics_metric_contract(
    row: Mapping[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate row-level metric/provenance invariants used by table evidence.

    The validator is intentionally conservative and data-only.  It catches
    impossible benchmark rows such as exact-reference self-comparisons with
    nonzero observable/fidelity errors, and it verifies that table fields mirror
    the metrics/resources they claim to summarize.  When ``strict`` is true a
    failed validation raises ``ValueError``; otherwise a structured status
    payload is returned for table-level quarantine/reporting.
    """

    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    metrics = row_dict.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    resources = row_dict.get("resources", {})
    if not isinstance(resources, Mapping):
        resources = {}
    table_fields = row_dict.get("table_fields", {})
    if not isinstance(table_fields, Mapping):
        table_fields = {}
    provenance = row_dict.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}

    violations: list[str] = []
    status = str(row_dict.get("status", ""))
    if status not in DYNAMICS_ROW_STATUS_VALUES:
        violations.append(f"unsupported_status:{status}")

    for key in _METRIC_TABLE_MIRROR_KEYS:
        if key in table_fields and key in metrics:
            if not _values_match(table_fields.get(key), metrics.get(key)):
                violations.append(f"table_field_metric_mismatch:{key}")
    for key in _RESOURCE_TABLE_MIRROR_KEYS:
        if key in table_fields and key in resources:
            if not _values_match(table_fields.get(key), resources.get(key), atol=0.0):
                violations.append(f"table_field_resource_mismatch:{key}")

    algorithm_id = str(row_dict.get("algorithm_id", ""))
    if status == "completed" and algorithm_id == "dyn_exact_reference":
        if bool(row_dict.get("qpu_faithful", False)):
            violations.append("exact_reference_marked_qpu_faithful")
        if not bool(row_dict.get("exact_assisted", False)):
            violations.append("exact_reference_not_marked_exact_assisted")

        for key in (
            "mean_abs_energy_total_error",
            "max_abs_energy_total_error",
            "final_abs_energy_total_error",
        ):
            value = _finite_float_or_none(metrics.get(key))
            if value is not None and abs(value) > 1e-12:
                violations.append(f"exact_reference_nonzero_energy_error:{key}")

        obs_policy = str(metrics.get("epsilon_obs_2_policy", ""))
        obs_values = [
            metrics.get("mean_abs_primary_density_error"),
            metrics.get("max_abs_primary_density_error"),
            metrics.get("max_abs_site_occupations_error"),
            table_fields.get("epsilon_obs_2"),
        ]
        if obs_policy == "exact_reference_self_comparison":
            for idx, value in enumerate(obs_values):
                maybe = _finite_float_or_none(value)
                if maybe is not None and abs(maybe) > 1e-12:
                    violations.append(f"exact_reference_nonzero_observable_error:{idx}")

        fidelity_policy = str(metrics.get("fidelity_policy", ""))
        if fidelity_policy == "exact_reference_self_comparison":
            for key in ("one_minus_final_fidelity_exact", "one_minus_min_fidelity_exact"):
                value = _finite_float_or_none(metrics.get(key))
                if value is not None and abs(value) > 1e-12:
                    violations.append(f"exact_reference_nonzero_fidelity_error:{key}")
            for key in ("final_fidelity_exact", "min_fidelity_exact"):
                value = _finite_float_or_none(metrics.get(key))
                if value is not None and abs(value - 1.0) > 1e-12:
                    violations.append(f"exact_reference_fidelity_not_one:{key}")

    if bool(row_dict.get("qpu_faithful", False)):
        if provenance.get("exact_reference_controller_inputs") is True:
            violations.append("qpu_faithful_exact_reference_controller_inputs")
        if provenance.get("uses_reference_for_decision") is True:
            violations.append("qpu_faithful_uses_reference_for_decision")
        if provenance.get("uses_future_exact_forecast_for_decision") is True:
            violations.append("qpu_faithful_uses_future_exact_forecast_for_decision")

    result = {
        "schema": "dynamics_metric_contract_validation_v1",
        "passed": not violations,
        "violation_count": int(len(violations)),
        "violations": list(violations),
    }
    if strict and violations:
        joined = ", ".join(violations)
        raise ValueError(f"dynamics metric contract failed: {joined}")
    return json_safe(result)


@dataclass(frozen=True)
class DynamicsTableFields:
    mean_abs_energy_total_error: float | None = None
    epsilon_obs_2: float | None = None
    one_minus_min_fidelity_exact: float | None = None
    epsilon_spec: float | None = None
    compiled_count_2q_total: int | None = None
    compiled_depth_2q_total: int | None = None
    compiled_depth_total: int | None = None
    shots_total: int | None = None
    table_status_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = {key: getattr(self, key) for key in DYNAMICS_TABLE_FIELD_KEYS}
        return json_safe(out)


@dataclass(frozen=True)
class DynamicsBenchmarkCase:
    case_id: str
    family: str
    table_class: str
    artifact_json: str
    description: str = ""
    t_final: float = 0.2
    num_times: int = 5
    loader_mode: str = "replay_family"
    generator_family: str = "match_adapt"
    fallback_family: str = "full_meta"
    append_pool_family: str = "match_replay"
    tuning_class: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DynamicsBenchmarkCase":
        missing = [
            key
            for key in ("case_id", "family", "table_class", "artifact_json")
            if payload.get(key) in (None, "")
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"generic dynamics case requires explicit {joined}")
        return cls(
            case_id=str(payload["case_id"]),
            family=str(payload["family"]),
            table_class=str(payload["table_class"]),
            artifact_json=str(payload["artifact_json"]),
            description=str(payload.get("description", "")),
            t_final=float(payload.get("t_final", 0.2)),
            num_times=int(payload.get("num_times", 5)),
            loader_mode=str(payload.get("loader_mode", "replay_family")),
            generator_family=str(payload.get("generator_family", "match_adapt")),
            fallback_family=str(payload.get("fallback_family", "full_meta")),
            append_pool_family=str(payload.get("append_pool_family", "match_replay")),
            tuning_class=validate_dynamics_tuning_class(
                family=str(payload["family"]),
                tuning_class=payload.get("tuning_class", None),
            ),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def normalize_dynamics_tuning_class(value: Any | None) -> str | None:
    """Return the canonical coarse dynamics tuning class, accepting legacy aliases."""

    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return DYNAMICS_TUNING_CLASS_ALIASES.get(raw, raw)


def coarse_dynamics_tuning_class_for_family(family: str) -> str:
    """Return the canonical coarse dynamics tuning class for a problem family."""

    return DYNAMICS_COARSE_TUNING_CLASS_BY_FAMILY.get(str(family).strip(), "unclassified")


def validate_dynamics_tuning_class(*, family: str, tuning_class: Any | None) -> str | None:
    """Validate a manifest-provided coarse tuning class against family policy."""

    raw = normalize_dynamics_tuning_class(tuning_class)
    expected = coarse_dynamics_tuning_class_for_family(str(family))
    if not raw:
        return None if expected == "unclassified" else expected
    if raw not in DYNAMICS_COARSE_TUNING_CLASSES:
        known = ", ".join(DYNAMICS_COARSE_TUNING_CLASSES)
        raise ValueError(f"unsupported dynamics tuning_class {raw!r}; expected one of {known}")
    if expected != "unclassified" and raw != expected:
        raise ValueError(
            f"dynamics tuning_class {raw!r} does not match family {family!r} expected {expected!r}"
        )
    return raw


def dynamics_tuning_class(case: DynamicsBenchmarkCase) -> str:
    """Return the coarse Hamiltonian-class key used for class tuning."""

    explicit = validate_dynamics_tuning_class(
        family=str(case.family),
        tuning_class=case.tuning_class,
    )
    if explicit:
        return explicit
    return coarse_dynamics_tuning_class_for_family(str(case.family))


def _settings_digest(payload: Mapping[str, Any]) -> str:
    text = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def build_dynamics_tuning_provenance(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    settings_kind: str,
    settings_payload: Mapping[str, Any] | None = None,
    settings_source: str = DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE,
    variant_id: str | None = None,
    locked: bool = False,
) -> dict[str, Any]:
    """Build additive tuning provenance for dynamics benchmark rows.

    The settings ID is deliberately class-scoped.  It may depend on the
    algorithm, coarse tuning class, variant, and normalized settings knobs,
    but not on a specific Hamiltonian fixture, artifact path, time grid, or
    command line.
    """

    raw_settings = dict(settings_payload or {})
    forbidden = sorted(set(raw_settings) & _FORBIDDEN_TUNING_ID_KEYS)
    if forbidden:
        joined = ", ".join(forbidden)
        raise ValueError(f"dynamics tuning settings ID payload contains case-specific keys: {joined}")

    tuning_class = dynamics_tuning_class(case)
    id_payload = {
        "algorithm_id": str(algorithm_id),
        "settings_kind": str(settings_kind),
        "settings_payload": raw_settings,
        "settings_source": str(settings_source),
        "tuning_class": str(tuning_class),
        "variant_id": None if variant_id is None else str(variant_id),
    }
    digest = _settings_digest(id_payload)
    settings_id = (
        f"{settings_source}:{tuning_class}:{algorithm_id}:"
        f"{settings_kind}{'' if variant_id is None else ':' + str(variant_id)}:{digest}"
    )
    is_controller = str(settings_kind) in {"controller", "mclachlan"}
    validation = "locked_coarse_class_tuned" if locked else "provisional_unlocked_coarse_class"
    if str(settings_source) == DYNAMICS_CASE_METADATA_OVERRIDE_SOURCE:
        validation = "provisional_case_metadata_override"
    elif str(settings_source) in {
        DYNAMICS_LEGACY_MISSING_TUNING_SOURCE,
        DYNAMICS_SKIPPED_TUNING_SOURCE,
        DYNAMICS_HH_LEGACY_TUNING_SOURCE,
    }:
        validation = "provisional_legacy_or_not_run"

    return json_safe(
        {
            "tuning_schema": DYNAMICS_TUNING_PROVENANCE_SCHEMA,
            "tuning_granularity": DYNAMICS_TUNING_GRANULARITY_CLASS,
            "tuning_class": str(tuning_class),
            "tuning_class_source": DYNAMICS_TUNING_CLASS_SOURCE,
            "source_table_class": str(case.table_class),
            "algorithm_id": str(algorithm_id),
            "variant_id": None if variant_id is None else str(variant_id),
            "settings_kind": str(settings_kind),
            "settings_source": str(settings_source),
            "settings_id": str(settings_id),
            "controller_settings_id": str(settings_id) if is_controller else None,
            "comparator_settings_id": None if is_controller else str(settings_id),
            "static_scaffold_scope": DYNAMICS_STATIC_SCAFFOLD_SCOPE_BENCHMARK_POINT,
            "static_scaffold_source": str(case.artifact_json),
            "class_tuned_result_locked": bool(locked),
            "tuning_validation_status": validation,
            "settings_payload_keys": sorted(str(key) for key in raw_settings),
        }
    )


@dataclass(frozen=True)
class DynamicsBenchmarkRow:
    family: str
    table_class: str
    case_id: str
    algorithm_id: str
    method_label: str
    status: str
    reason: str
    qpu_faithful: bool
    exact_assisted: bool
    diagnostic: bool
    artifact_json: str | None
    metrics: Mapping[str, Any]
    resources: Mapping[str, Any]
    provenance: Mapping[str, Any]
    table_fields: DynamicsTableFields | Mapping[str, Any]
    schema: str = DYNAMICS_BENCHMARK_ROW_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        if self.status not in DYNAMICS_ROW_STATUS_VALUES:
            raise ValueError(
                f"unsupported dynamics row status {self.status!r}; "
                f"expected one of {DYNAMICS_ROW_STATUS_VALUES}"
            )
        table_fields = (
            self.table_fields.to_dict()
            if isinstance(self.table_fields, DynamicsTableFields)
            else {key: self.table_fields.get(key) for key in DYNAMICS_TABLE_FIELD_KEYS}
        )
        return json_safe(
            {
                "schema": self.schema,
                "family": self.family,
                "table_class": self.table_class,
                "case_id": self.case_id,
                "algorithm_id": self.algorithm_id,
                "method_label": self.method_label,
                "status": self.status,
                "reason": self.reason,
                "qpu_faithful": bool(self.qpu_faithful),
                "exact_assisted": bool(self.exact_assisted),
                "diagnostic": bool(self.diagnostic),
                "artifact_json": self.artifact_json,
                "metrics": dict(self.metrics),
                "resources": dict(self.resources),
                "provenance": dict(self.provenance),
                "table_fields": table_fields,
            }
        )


def dynamics_table_bundle_payload(
    *,
    rows: Sequence[DynamicsBenchmarkRow | Mapping[str, Any]],
    label: str = "generic_dynamics_benchmark",
) -> dict[str, Any]:
    row_dicts = [row.to_dict() if isinstance(row, DynamicsBenchmarkRow) else dict(row) for row in rows]
    validations = [validate_dynamics_metric_contract(row, strict=False) for row in row_dicts]
    status_counts: dict[str, int] = {}
    for row in row_dicts:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return json_safe(
        {
            "schema": DYNAMICS_TABLE_BUNDLE_SCHEMA,
            "label": str(label),
            "row_count": int(len(row_dicts)),
            "status_counts": dict(sorted(status_counts.items())),
            "table_field_keys": list(DYNAMICS_TABLE_FIELD_KEYS),
            "metric_contract_validation": {
                "schema": "dynamics_metric_contract_validation_summary_v1",
                "row_count": int(len(validations)),
                "passed_count": int(sum(1 for item in validations if item.get("passed"))),
                "failed_count": int(sum(1 for item in validations if not item.get("passed"))),
                "failed_rows": [
                    {
                        "index": int(index),
                        "algorithm_id": row_dicts[index].get("algorithm_id"),
                        "case_id": row_dicts[index].get("case_id"),
                        "violations": item.get("violations", []),
                    }
                    for index, item in enumerate(validations)
                    if not item.get("passed")
                ],
            },
            "rows": row_dicts,
        }
    )


def table_iv_prune_pilot_contract() -> dict[str, Any]:
    """Return the explicit single-pair prune pilot contract for Table IV.

    This is not an aggregation helper and must not be treated as class-wide
    evidence.  It freezes the pilot label and paired deltas so later table code
    cannot silently promote it into medians.
    """

    deltas = {
        key: float(_PRUNE_ENABLED_PILOT[key]) - float(_NO_PRUNE_PILOT[key])
        for key in (
            "mean_abs_energy_total_error",
            "primary_observable_mae_span",
            "prune_count",
            "final_params",
        )
    }
    return json_safe(
        {
            "aggregation_scope": TABLE_IV_PRUNE_PILOT_AGGREGATION_SCOPE,
            "status": TABLE_IV_PRUNE_PILOT_STATUS,
            "paired_case_count": 1,
            "class_wide_evidence": False,
            "prune_enabled": dict(_PRUNE_ENABLED_PILOT),
            "no_prune": dict(_NO_PRUNE_PILOT),
            "delta_prune_minus_no_prune": deltas,
        }
    )


__all__ = [
    "DYNAMICS_BENCHMARK_ROW_SCHEMA",
    "DYNAMICS_CANONICAL_CONTROLLER_ALGORITHM_ID",
    "DYNAMICS_CANONICAL_CONTROLLER_VARIANT_ID",
    "DYNAMICS_CLASS_SETTINGS_KINDS",
    "DYNAMICS_COARSE_TUNING_CLASS_BY_FAMILY",
    "DYNAMICS_COARSE_TUNING_CLASSES",
    "DYNAMICS_TUNING_CLASS_ALIASES",
    "DYNAMICS_FORBIDDEN_TUNING_ID_KEYS",
    "DYNAMICS_ROW_STATUS_VALUES",
    "DYNAMICS_TABLE_I_ALGORITHMS",
    "DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND",
    "QISKIT_COMMUNITY_DYNAMICS_ALGORITHMS",
    "QISKIT_COMMUNITY_DYNAMICS_ALGORITHM_SETTINGS_KIND",
    "DYNAMICS_ALL_ALGORITHM_CLASS_SETTINGS_CANDIDATE_SOURCE",
    "DYNAMICS_SETTINGS_KIND_BENCHMARK",
    "DYNAMICS_SETTINGS_KIND_COMPARATOR",
    "DYNAMICS_SETTINGS_KIND_CONTROLLER",
    "DYNAMICS_SETTINGS_KIND_MCLACHLAN",
    "DYNAMICS_SETTINGS_KIND_REFERENCE",
    "DYNAMICS_TABLE_BUNDLE_SCHEMA",
    "DYNAMICS_TABLE_FIELD_KEYS",
    "DynamicsBenchmarkCase",
    "DynamicsBenchmarkRow",
    "DynamicsTableFields",
    "TABLE_IV_PRUNE_PILOT_AGGREGATION_SCOPE",
    "TABLE_IV_PRUNE_PILOT_STATUS",
    "coarse_dynamics_tuning_class_for_family",
    "dynamics_tuning_class",
    "dynamics_table_bundle_payload",
    "json_safe",
    "normalize_dynamics_tuning_class",
    "table_iv_prune_pilot_contract",
    "validate_dynamics_metric_contract",
    "validate_dynamics_tuning_class",
]
