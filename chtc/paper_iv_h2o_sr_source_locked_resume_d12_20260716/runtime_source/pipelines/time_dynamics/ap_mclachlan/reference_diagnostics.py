"""Report-only cached reference energy utilities for AP-McLachlan runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REFERENCE_ENERGY_TRAJECTORY_SCHEMA_V1 = "reference_energy_trajectory_v1"
REFERENCE_TIME_MATCH_EXACT_V1 = "exact_time_match_v1"


@dataclass(frozen=True)
class ReferenceEnergyPoint:
    """One cached reference energy value for reporting."""

    time: float
    energy: float
    source: str = "cached_reference_energy"
    observables: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.time)):
            raise ValueError("Reference energy time must be finite.")
        if not np.isfinite(float(self.energy)):
            raise ValueError("Reference energy must be finite.")
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "energy", float(self.energy))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "observables", dict(self.observables or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "time": float(self.time),
            "energy": float(self.energy),
            "source": str(self.source),
            "observables": _json_safe(dict(self.observables or {})),
            "metadata": _json_safe(dict(self.metadata or {})),
        }


@dataclass(frozen=True)
class ReferenceEnergyTrajectory:
    """Cached reference energy trajectory for reporting."""

    points: tuple[ReferenceEnergyPoint, ...]
    source: str = "cached_reference_energy_trajectory"
    match_kind: str = REFERENCE_TIME_MATCH_EXACT_V1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = tuple(self.points)
        times = [float(point.time) for point in points]
        if len(set(times)) != len(times):
            raise ValueError("Reference energy trajectory contains duplicate times.")
        if str(self.match_kind) != REFERENCE_TIME_MATCH_EXACT_V1:
            raise ValueError(
                "Unsupported reference energy match_kind "
                f"{self.match_kind!r}; only {REFERENCE_TIME_MATCH_EXACT_V1!r} is implemented."
            )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "match_kind", str(self.match_kind))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": REFERENCE_ENERGY_TRAJECTORY_SCHEMA_V1,
            "source": str(self.source),
            "match_kind": str(self.match_kind),
            "point_count": int(len(self.points)),
            "points": [point.to_json_dict() for point in self.points],
            "metadata": _json_safe(dict(self.metadata or {})),
        }


def load_reference_energy_trajectory(path: str | Path) -> ReferenceEnergyTrajectory:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Reference energy JSON must be an object: {path}")
    return reference_energy_trajectory_from_payload(payload)


def reference_energy_trajectory_from_payload(
    payload: Mapping[str, Any],
) -> ReferenceEnergyTrajectory:
    schema = payload.get("schema")
    if schema not in {None, "", REFERENCE_ENERGY_TRAJECTORY_SCHEMA_V1}:
        raise ValueError(
            "Unsupported reference energy schema "
            f"{schema!r}; expected {REFERENCE_ENERGY_TRAJECTORY_SCHEMA_V1!r}."
        )
    points_raw = payload.get("points")
    if not isinstance(points_raw, Sequence) or isinstance(points_raw, (str, bytes)):
        raise ValueError("Reference energy payload requires a sequence field `points`.")
    source = str(payload.get("source", "cached_reference_energy_trajectory"))
    points: list[ReferenceEnergyPoint] = []
    for index, item in enumerate(points_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"Reference energy point {index} must be an object.")
        point_source = str(item.get("source", source))
        energy = item.get("energy", item.get("reference_energy"))
        if energy is None:
            raise ValueError(f"Reference energy point {index} is missing `energy`.")
        if "time" not in item:
            raise ValueError(f"Reference energy point {index} is missing `time`.")
        points.append(
            ReferenceEnergyPoint(
                time=float(item["time"]),
                energy=float(energy),
                source=point_source,
                observables=_observables_from_reference_point(item),
                metadata=dict(item.get("metadata", {}) or {}),
            )
        )
    return ReferenceEnergyTrajectory(
        points=tuple(points),
        source=source,
        match_kind=str(payload.get("match_kind", REFERENCE_TIME_MATCH_EXACT_V1)),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def attach_reference_energy_diagnostics(
    *,
    plot_rows: Sequence[Mapping[str, Any]],
    reference: ReferenceEnergyTrajectory | None,
    atol: float = 1.0e-12,
) -> list[dict[str, Any]]:
    """Attach diagnostic energy errors to completed rows.

    Reference values are report-only. This function is intentionally runner-side
    plumbing and must not be called by controller selection or integrator code.
    """

    return attach_reference_energy_diagnostics_with_prefix(
        plot_rows=plot_rows,
        reference=reference,
        atol=float(atol),
        field_prefix="",
    )


def attach_reference_energy_diagnostics_with_prefix(
    *,
    plot_rows: Sequence[Mapping[str, Any]],
    reference: ReferenceEnergyTrajectory | None,
    atol: float = 1.0e-12,
    field_prefix: str = "",
) -> list[dict[str, Any]]:
    """Attach report-only reference diagnostics with optional field prefix."""

    prefix = str(field_prefix)
    rows = [dict(row) for row in plot_rows]
    if reference is None:
        for row in rows:
            row.setdefault(_field(prefix, "reference_energy"), None)
            row.setdefault(_field(prefix, "energy_error"), None)
            row.setdefault(_field(prefix, "abs_energy_error"), None)
            row.setdefault(
                _field(prefix, "reference_energy_missing_reason"),
                "reference_not_provided",
            )
        return rows
    if float(atol) < 0.0 or not np.isfinite(float(atol)):
        raise ValueError("reference energy time tolerance must be finite and non-negative.")
    for row in rows:
        matched, time_delta = _match_reference_point(
            float(row["time"]),
            reference=reference,
            atol=float(atol),
        )
        if matched is None:
            row[_field(prefix, "reference_energy")] = None
            row[_field(prefix, "energy_error")] = None
            row[_field(prefix, "abs_energy_error")] = None
            _attach_missing_reference_observables(
                row,
                reason="no_time_match",
                field_prefix=prefix,
            )
            row[_field(prefix, "reference_energy_source")] = str(reference.source)
            row[_field(prefix, "reference_time_match_kind")] = str(reference.match_kind)
            row[_field(prefix, "reference_time_delta")] = None
            row[_field(prefix, "reference_energy_missing_reason")] = "no_time_match"
            continue
        energy = float(row["energy_expectation"])
        error = float(energy - float(matched.energy))
        row[_field(prefix, "reference_energy")] = float(matched.energy)
        row[_field(prefix, "reference_energy_source")] = str(matched.source)
        row[_field(prefix, "reference_time_match_kind")] = str(reference.match_kind)
        row[_field(prefix, "reference_time_delta")] = float(time_delta)
        row[_field(prefix, "reference_energy_missing_reason")] = None
        row[_field(prefix, "energy_error")] = float(error)
        row[_field(prefix, "abs_energy_error")] = float(abs(error))
        _attach_matched_reference_observables(row, matched, field_prefix=prefix)
    return rows


def reference_energy_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    field_prefix: str = "",
    summary_prefix: str = "",
) -> dict[str, Any]:
    prefix = str(field_prefix)
    out_prefix = str(summary_prefix)
    missing_key = _field(prefix, "reference_energy_missing_reason")
    energy_key = _field(prefix, "reference_energy")
    source_key = _field(prefix, "reference_energy_source")
    reference_provided = any(
        (missing_key in row and row.get(missing_key) != "reference_not_provided")
        or row.get(energy_key) is not None
        or row.get(source_key) is not None
        for row in rows
    )
    matched = [
        row for row in rows
        if row.get(_field(prefix, "reference_energy")) is not None
        and row.get(_field(prefix, "abs_energy_error")) is not None
    ]
    unmatched_count = sum(
        1
        for row in rows
        if row.get(_field(prefix, "reference_energy_missing_reason")) == "no_time_match"
    )
    errors = [float(row[_field(prefix, "abs_energy_error")]) for row in matched]
    final_abs_error = None
    if rows and rows[-1].get(_field(prefix, "abs_energy_error")) is not None:
        final_abs_error = float(rows[-1][_field(prefix, "abs_energy_error")])
    summary = {
        _summary_field(out_prefix, "reference_energy_diagnostics_enabled"): bool(matched),
        _summary_field(out_prefix, "reference_energy_reference_provided"): bool(reference_provided),
        _summary_field(out_prefix, "reference_energy_matched_count"): int(len(matched)),
        _summary_field(out_prefix, "reference_energy_unmatched_count"): int(unmatched_count),
        _summary_field(out_prefix, "max_abs_energy_error"): None if not errors else float(max(errors)),
        _summary_field(out_prefix, "final_abs_energy_error"): final_abs_error,
    }
    summary.update(
        _reference_observable_summary(
            rows,
            field_prefix=prefix,
            summary_prefix=out_prefix,
        )
    )
    return summary


def _observables_from_reference_point(item: Mapping[str, Any]) -> dict[str, Any]:
    observables = dict(item.get("observables", {}) or {})
    for key in (
        "observable_schema",
        "observable_telemetry_supported",
        "observable_family",
        "n_up_site",
        "n_dn_site",
        "site_occupations",
        "site_occupations_up",
        "site_occupations_dn",
        "site_occupations_label",
        "site_occupations_component_labels",
        "doublon",
        "staggered",
        "primary_density",
        "observable_evaluation_policy",
    ):
        if key in item and key not in observables:
            observables[key] = item[key]
    return observables


def _attach_missing_reference_observables(
    row: dict[str, Any],
    *,
    reason: str,
    field_prefix: str,
) -> None:
    prefix = str(field_prefix)
    if "doublon" in row:
        row[_field(prefix, "doublon_exact")] = None
        row[_field(prefix, "abs_doublon_error")] = None
    if "primary_density" in row:
        row[_field(prefix, "primary_density_exact")] = None
        row[_field(prefix, "abs_primary_density_error")] = None
    if "staggered" in row:
        row[_field(prefix, "staggered_exact")] = None
        row[_field(prefix, "abs_staggered_error")] = None
    if "site_occupations" in row:
        row[_field(prefix, "site_occupations_exact")] = None
        row[_field(prefix, "site_occupations_abs_error")] = None
        row[_field(prefix, "site_occupations_abs_error_max")] = None
    if "n_up_site" in row:
        row[_field(prefix, "n_up_site_exact")] = None
        row[_field(prefix, "site_occupations_up_exact")] = None
    if "n_dn_site" in row:
        row[_field(prefix, "n_dn_site_exact")] = None
        row[_field(prefix, "site_occupations_dn_exact")] = None
    if any(key in row for key in ("doublon", "site_occupations", "primary_density")):
        row[_field(prefix, "reference_observables_missing_reason")] = str(reason)


def _attach_matched_reference_observables(
    row: dict[str, Any],
    matched: ReferenceEnergyPoint,
    *,
    field_prefix: str,
) -> None:
    prefix = str(field_prefix)
    observables = dict(matched.observables or {})
    if not observables:
        _attach_missing_reference_observables(
            row,
            reason="reference_observables_not_provided",
            field_prefix=prefix,
        )
        return
    if "n_up_site" in observables:
        value = _float_list_or_none(observables.get("n_up_site"))
        row[_field(prefix, "n_up_site_exact")] = value
        row[_field(prefix, "site_occupations_up_exact")] = value
    if "n_dn_site" in observables:
        value = _float_list_or_none(observables.get("n_dn_site"))
        row[_field(prefix, "n_dn_site_exact")] = value
        row[_field(prefix, "site_occupations_dn_exact")] = value
    if "site_occupations" in observables:
        exact_sites = _float_list_or_none(observables.get("site_occupations"))
        row[_field(prefix, "site_occupations_exact")] = exact_sites
        trial_sites = _float_list_or_none(row.get("site_occupations"))
        if exact_sites is not None and trial_sites is not None and len(exact_sites) == len(trial_sites):
            errors = [float(abs(float(a) - float(b))) for a, b in zip(trial_sites, exact_sites)]
            row[_field(prefix, "site_occupations_abs_error")] = errors
            row[_field(prefix, "site_occupations_abs_error_max")] = (
                None if not errors else float(max(errors))
            )
        else:
            row[_field(prefix, "site_occupations_abs_error")] = None
            row[_field(prefix, "site_occupations_abs_error_max")] = None
    for key, exact_key, error_key in (
        ("doublon", "doublon_exact", "abs_doublon_error"),
        ("primary_density", "primary_density_exact", "abs_primary_density_error"),
        ("staggered", "staggered_exact", "abs_staggered_error"),
    ):
        exact_value = _finite_or_none(observables.get(key))
        row[_field(prefix, exact_key)] = exact_value
        trial_value = _finite_or_none(row.get(key))
        row[_field(prefix, error_key)] = (
            None
            if exact_value is None or trial_value is None
            else float(abs(float(trial_value) - float(exact_value)))
        )
    row[_field(prefix, "reference_observables_source")] = str(matched.source)
    row[_field(prefix, "reference_observables_missing_reason")] = None


def _reference_observable_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    field_prefix: str,
    summary_prefix: str,
) -> dict[str, Any]:
    prefix = str(field_prefix)
    out_prefix = str(summary_prefix)
    doublon_errors = _finite_row_values(rows, _field(prefix, "abs_doublon_error"))
    site_errors = _finite_row_values(rows, _field(prefix, "site_occupations_abs_error_max"))
    primary_errors = _finite_row_values(rows, _field(prefix, "abs_primary_density_error"))
    final_row = rows[-1] if rows else {}
    return {
        _summary_field(out_prefix, "reference_observable_diagnostics_enabled"): bool(
            doublon_errors or site_errors or primary_errors
        ),
        _summary_field(out_prefix, "final_doublon"): _finite_or_none(final_row.get("doublon")),
        _summary_field(out_prefix, "final_doublon_exact"): _finite_or_none(
            final_row.get(_field(prefix, "doublon_exact"))
        ),
        _summary_field(out_prefix, "final_abs_doublon_error"): _finite_or_none(
            final_row.get(_field(prefix, "abs_doublon_error"))
        ),
        _summary_field(out_prefix, "max_abs_doublon_error"): None if not doublon_errors else float(max(doublon_errors)),
        _summary_field(out_prefix, "mean_abs_doublon_error"): (
            None if not doublon_errors else float(np.mean(np.asarray(doublon_errors, dtype=float)))
        ),
        _summary_field(out_prefix, "final_site_occupations"): _float_list_or_none(final_row.get("site_occupations")),
        _summary_field(out_prefix, "final_site_occupations_exact"): _float_list_or_none(
            final_row.get(_field(prefix, "site_occupations_exact"))
        ),
        _summary_field(out_prefix, "final_site_occupations_abs_error_max"): _finite_or_none(
            final_row.get(_field(prefix, "site_occupations_abs_error_max"))
        ),
        _summary_field(out_prefix, "max_abs_site_occupations_error"): None if not site_errors else float(max(site_errors)),
        _summary_field(out_prefix, "mean_abs_site_occupations_error"): (
            None if not site_errors else float(np.mean(np.asarray(site_errors, dtype=float)))
        ),
        _summary_field(out_prefix, "final_primary_density"): _finite_or_none(final_row.get("primary_density")),
        _summary_field(out_prefix, "final_primary_density_exact"): _finite_or_none(
            final_row.get(_field(prefix, "primary_density_exact"))
        ),
        _summary_field(out_prefix, "final_abs_primary_density_error"): _finite_or_none(
            final_row.get(_field(prefix, "abs_primary_density_error"))
        ),
        _summary_field(out_prefix, "max_abs_primary_density_error"): (
            None if not primary_errors else float(max(primary_errors))
        ),
    }


def _field(prefix: str, name: str) -> str:
    return f"{str(prefix)}{str(name)}"


def _summary_field(prefix: str, name: str) -> str:
    return f"{str(prefix)}{str(name)}"


def _finite_row_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _finite_or_none(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _float_list_or_none(value: Any) -> list[float] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return [float(x) for x in arr.tolist()]


def _match_reference_point(
    time_value: float,
    *,
    reference: ReferenceEnergyTrajectory,
    atol: float,
) -> tuple[ReferenceEnergyPoint | None, float | None]:
    best_point: ReferenceEnergyPoint | None = None
    best_delta: float | None = None
    for point in reference.points:
        delta = abs(float(point.time) - float(time_value))
        if best_delta is None or delta < best_delta:
            best_delta = float(delta)
            best_point = point
    if best_point is None or best_delta is None or best_delta > float(atol):
        return None, None
    return best_point, float(best_delta)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    return out if np.isfinite(out) else None


__all__ = [
    "REFERENCE_ENERGY_TRAJECTORY_SCHEMA_V1",
    "REFERENCE_TIME_MATCH_EXACT_V1",
    "ReferenceEnergyPoint",
    "ReferenceEnergyTrajectory",
    "attach_reference_energy_diagnostics",
    "attach_reference_energy_diagnostics_with_prefix",
    "load_reference_energy_trajectory",
    "reference_energy_summary",
    "reference_energy_trajectory_from_payload",
]
