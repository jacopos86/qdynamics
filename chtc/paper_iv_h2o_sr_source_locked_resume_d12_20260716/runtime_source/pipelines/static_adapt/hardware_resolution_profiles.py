"""Named hardware-resolution profile resolver for static ADAPT.

This module is intentionally a thin fail-closed wrapper around the existing
``ideal``/``manual`` gradient floor plumbing.  Named profiles only provide an
audited source for scalar gradient floors; scoring layers continue to consume
validated effective ``ideal`` or ``manual`` settings.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipelines.static_adapt.resume_scaffold import digest_jsonable, file_sha256

HARDWARE_RESOLUTION_PROFILE_MANIFEST_SCHEMA = "hardware_resolution_profile_manifest_v1"
HARDWARE_RESOLUTION_GRADIENT_PROFILE_SCHEMA = "hardware_resolution_gradient_profile_v1"
HARDWARE_RESOLUTION_PROFILE_UNITS = "energy_gradient_abs"

_HARDWARE_RESOLUTION_RUNTIME_SCHEMA = "gradient_resolution_v1"
_VALID_REQUESTED_MODES = {"ideal", "manual", "profile"}
_VALID_EFFECTIVE_MODES = {"ideal", "manual"}


@dataclass(frozen=True)
class ResolvedHardwareResolutionConfig:
    """Resolved runtime hardware-resolution floor configuration."""

    requested_mode: str
    effective_mode: str
    gradient_hw_floor: float
    gradient_drift_floor: float
    floor_source: str
    profile_name: str | None = None
    profile_json: str | None = None
    profile_json_sha256: str | None = None
    profile_manifest_digest: str | None = None
    profile_digest: str | None = None
    profile_manifest_schema: str | None = None
    profile_schema: str | None = None
    profile_units: str | None = None
    profile_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        requested = str(self.requested_mode).strip().lower()
        effective = str(self.effective_mode).strip().lower()
        if requested not in _VALID_REQUESTED_MODES:
            raise ValueError(
                "requested hardware_resolution_mode must be one of "
                "{'ideal','manual','profile'}."
            )
        if effective not in _VALID_EFFECTIVE_MODES:
            raise ValueError("effective hardware_resolution_mode must be one of {'ideal','manual'}.")
        hw = float(self.gradient_hw_floor)
        drift = float(self.gradient_drift_floor)
        if (not math.isfinite(hw)) or hw < 0.0:
            raise ValueError("gradient_hw_floor must be finite and nonnegative.")
        if (not math.isfinite(drift)) or drift < 0.0:
            raise ValueError("gradient_drift_floor must be finite and nonnegative.")
        object.__setattr__(self, "requested_mode", requested)
        object.__setattr__(self, "effective_mode", effective)
        object.__setattr__(self, "gradient_hw_floor", float(hw))
        object.__setattr__(self, "gradient_drift_floor", float(drift))
        object.__setattr__(self, "floor_source", str(self.floor_source))

    def to_telemetry(self) -> dict[str, Any]:
        """Return additive artifact telemetry for ``continuation.hardware_resolution``."""

        profile_provenance = (
            None if self.profile_provenance is None else dict(self.profile_provenance)
        )
        return {
            "schema": _HARDWARE_RESOLUTION_RUNTIME_SCHEMA,
            "mode": str(self.effective_mode),
            "mode_requested": str(self.requested_mode),
            "mode_effective": str(self.effective_mode),
            "gradient_hw_floor": float(self.gradient_hw_floor),
            "gradient_drift_floor": float(self.gradient_drift_floor),
            "floor_source": str(self.floor_source),
            "default_floor_source": str(self.floor_source),
            "profile_name": self.profile_name,
            "profile_json": self.profile_json,
            "profile_json_path": self.profile_json,
            "profile_json_sha256": self.profile_json_sha256,
            "profile_manifest_digest": self.profile_manifest_digest,
            "profile_digest": self.profile_digest,
            "profile_manifest_schema": self.profile_manifest_schema,
            "profile_schema": self.profile_schema,
            "profile_units": self.profile_units,
            "profile_provenance": profile_provenance,
        }


def _arg_present(raw: Any) -> bool:
    if raw is None:
        return False
    return str(raw).strip() != ""


def _coerce_cli_floor(raw: Any, *, label: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"{label} must be finite and nonnegative.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and nonnegative.") from exc
    if (not math.isfinite(value)) or value < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative.")
    return float(value)


def _require_zero_cli_floors(*, gradient_hw_floor: float, gradient_drift_floor: float, mode: str) -> None:
    if float(gradient_hw_floor) != 0.0 or float(gradient_drift_floor) != 0.0:
        raise ValueError(
            f"hardware_resolution_mode='{mode}' requires zero scalar gradient hardware/drift floors."
        )


def _nonempty_profile_name(raw: Any) -> str:
    if raw is None:
        raise ValueError("hardware_resolution_profile_name is required for profile mode.")
    name = str(raw).strip()
    if name == "":
        raise ValueError("hardware_resolution_profile_name is required for profile mode.")
    return name


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Hardware-resolution profile JSON is not valid JSON: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read hardware-resolution profile JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Hardware-resolution profile manifest must be a JSON object.")
    return dict(payload)


def _json_floor(profile: Mapping[str, Any], *, key: str) -> float:
    if key not in profile:
        raise ValueError(f"Hardware-resolution profile is missing required floor {key!r}.")
    raw = profile[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"Hardware-resolution profile floor {key!r} must be a JSON number.")
    value = float(raw)
    if (not math.isfinite(value)) or value < 0.0:
        raise ValueError(
            f"Hardware-resolution profile floor {key!r} must be finite and nonnegative."
        )
    return float(value)


def _resolve_profile_mode(
    *,
    gradient_hw_floor: float,
    gradient_drift_floor: float,
    profile_json: str | Path | None,
    profile_name: str | None,
) -> ResolvedHardwareResolutionConfig:
    _require_zero_cli_floors(
        gradient_hw_floor=float(gradient_hw_floor),
        gradient_drift_floor=float(gradient_drift_floor),
        mode="profile",
    )
    if not _arg_present(profile_json):
        raise ValueError("hardware_resolution_profile_json is required for profile mode.")
    requested_name = _nonempty_profile_name(profile_name)
    json_path = Path(str(profile_json)).expanduser()
    manifest = _load_manifest(json_path)
    manifest_schema = manifest.get("schema")
    if manifest_schema != HARDWARE_RESOLUTION_PROFILE_MANIFEST_SCHEMA:
        raise ValueError(
            "Hardware-resolution profile manifest schema must be "
            f"{HARDWARE_RESOLUTION_PROFILE_MANIFEST_SCHEMA!r}."
        )
    profiles = manifest.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("Hardware-resolution profile manifest must contain a profiles object.")
    if requested_name not in profiles:
        raise ValueError(f"Hardware-resolution profile {requested_name!r} was not found in manifest.")
    selected_raw = profiles[requested_name]
    if not isinstance(selected_raw, Mapping):
        raise ValueError(f"Hardware-resolution profile {requested_name!r} must be a JSON object.")
    selected = dict(selected_raw)
    profile_schema = selected.get("schema")
    if profile_schema != HARDWARE_RESOLUTION_GRADIENT_PROFILE_SCHEMA:
        raise ValueError(
            "Hardware-resolution gradient profile schema must be "
            f"{HARDWARE_RESOLUTION_GRADIENT_PROFILE_SCHEMA!r}."
        )
    if selected.get("name") != requested_name:
        raise ValueError(
            f"Hardware-resolution profile name mismatch for {requested_name!r}."
        )
    units = selected.get("units")
    if units != HARDWARE_RESOLUTION_PROFILE_UNITS:
        raise ValueError(
            "Hardware-resolution profile units must be "
            f"{HARDWARE_RESOLUTION_PROFILE_UNITS!r}."
        )
    provenance_raw = selected.get("provenance", {})
    if provenance_raw is None:
        provenance_raw = {}
    if not isinstance(provenance_raw, Mapping):
        raise ValueError("Hardware-resolution profile provenance must be a JSON object when provided.")
    hw_floor = _json_floor(selected, key="gradient_hw_floor")
    drift_floor = _json_floor(selected, key="gradient_drift_floor")
    return ResolvedHardwareResolutionConfig(
        requested_mode="profile",
        effective_mode="manual",
        gradient_hw_floor=float(hw_floor),
        gradient_drift_floor=float(drift_floor),
        floor_source="profile_manifest",
        profile_name=str(requested_name),
        profile_json=str(json_path),
        profile_json_sha256=file_sha256(json_path),
        profile_manifest_digest=digest_jsonable(manifest),
        profile_digest=digest_jsonable({"name": requested_name, "profile": selected}),
        profile_manifest_schema=str(manifest_schema),
        profile_schema=str(profile_schema),
        profile_units=str(units),
        profile_provenance=dict(provenance_raw),
    )


def resolve_hardware_resolution_config(
    *,
    mode: Any,
    gradient_hw_floor: Any,
    gradient_drift_floor: Any,
    profile_json: str | Path | None = None,
    profile_name: str | None = None,
) -> ResolvedHardwareResolutionConfig:
    """Resolve requested hardware-resolution mode into scorer-ready settings.

    ``profile`` mode is provenance-only at this layer: it loads and validates an
    explicit named profile, then maps its floors into effective ``manual`` mode.
    """

    requested = str(mode or "ideal").strip().lower()
    if requested not in _VALID_REQUESTED_MODES:
        raise ValueError(
            "hardware_resolution_mode must be one of {'ideal','manual','profile'}."
        )
    hw_floor = _coerce_cli_floor(gradient_hw_floor, label="gradient_hw_floor")
    drift_floor = _coerce_cli_floor(gradient_drift_floor, label="gradient_drift_floor")
    has_profile_json = _arg_present(profile_json)
    has_profile_name = _arg_present(profile_name)

    if requested in {"ideal", "manual"} and (has_profile_json or has_profile_name):
        raise ValueError(
            f"hardware_resolution_mode='{requested}' rejects profile JSON/name arguments."
        )

    if requested == "ideal":
        _require_zero_cli_floors(
            gradient_hw_floor=float(hw_floor),
            gradient_drift_floor=float(drift_floor),
            mode="ideal",
        )
        return ResolvedHardwareResolutionConfig(
            requested_mode="ideal",
            effective_mode="ideal",
            gradient_hw_floor=0.0,
            gradient_drift_floor=0.0,
            floor_source="ideal_zero_floors",
        )

    if requested == "manual":
        return ResolvedHardwareResolutionConfig(
            requested_mode="manual",
            effective_mode="manual",
            gradient_hw_floor=float(hw_floor),
            gradient_drift_floor=float(drift_floor),
            floor_source="manual_scalar_floors",
        )

    return _resolve_profile_mode(
        gradient_hw_floor=float(hw_floor),
        gradient_drift_floor=float(drift_floor),
        profile_json=profile_json,
        profile_name=profile_name,
    )


__all__ = [
    "HARDWARE_RESOLUTION_GRADIENT_PROFILE_SCHEMA",
    "HARDWARE_RESOLUTION_PROFILE_MANIFEST_SCHEMA",
    "HARDWARE_RESOLUTION_PROFILE_UNITS",
    "ResolvedHardwareResolutionConfig",
    "resolve_hardware_resolution_config",
]
