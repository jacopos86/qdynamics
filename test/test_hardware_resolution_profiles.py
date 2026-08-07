from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.static_adapt.hardware_resolution_profiles import (
    HARDWARE_RESOLUTION_GRADIENT_PROFILE_SCHEMA,
    HARDWARE_RESOLUTION_PROFILE_MANIFEST_SCHEMA,
    HARDWARE_RESOLUTION_PROFILE_UNITS,
    resolve_hardware_resolution_config,
)
from pipelines.static_adapt.resume_scaffold import digest_jsonable, file_sha256


def _profile_manifest(
    *,
    name: str = "calib_a",
    hw_floor: object = 0.012,
    drift_floor: object = 0.003,
    manifest_schema: str = HARDWARE_RESOLUTION_PROFILE_MANIFEST_SCHEMA,
    profile_schema: str = HARDWARE_RESOLUTION_GRADIENT_PROFILE_SCHEMA,
    units: str = HARDWARE_RESOLUTION_PROFILE_UNITS,
    profile_name: object | None = "__same__",
    provenance: object = "__default__",
) -> dict[str, object]:
    selected: dict[str, object] = {
        "schema": profile_schema,
        "gradient_hw_floor": hw_floor,
        "gradient_drift_floor": drift_floor,
        "units": units,
    }
    if profile_name == "__same__":
        selected["name"] = name
    elif profile_name is not None:
        selected["name"] = profile_name
    if provenance == "__default__":
        selected["provenance"] = {
            "source": "unit-test-explicit-calibration",
            "generated_utc": "2026-05-16T00:00:00Z",
        }
    elif provenance is not None:
        selected["provenance"] = provenance
    return {
        "schema": manifest_schema,
        "profiles": {name: selected},
    }


def _write_manifest(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "hardware_profiles.json"
    path.write_text(json.dumps(payload, allow_nan=True, sort_keys=True), encoding="utf-8")
    return path


def test_profile_mode_resolves_named_profile_to_effective_manual_with_provenance(tmp_path: Path) -> None:
    manifest = _profile_manifest()
    path = _write_manifest(tmp_path, manifest)

    resolved = resolve_hardware_resolution_config(
        mode="profile",
        gradient_hw_floor=0.0,
        gradient_drift_floor=0.0,
        profile_json=path,
        profile_name="calib_a",
    )

    assert resolved.requested_mode == "profile"
    assert resolved.effective_mode == "manual"
    assert resolved.gradient_hw_floor == pytest.approx(0.012)
    assert resolved.gradient_drift_floor == pytest.approx(0.003)
    assert resolved.floor_source == "profile_manifest"
    assert resolved.profile_json == str(path)
    assert resolved.profile_json_sha256 == file_sha256(path)
    assert resolved.profile_manifest_digest == digest_jsonable(manifest)
    assert resolved.profile_digest == digest_jsonable(
        {"name": "calib_a", "profile": manifest["profiles"]["calib_a"]}
    )

    telemetry = resolved.to_telemetry()
    assert telemetry["schema"] == "gradient_resolution_v1"
    assert telemetry["mode"] == "manual"
    assert telemetry["mode_requested"] == "profile"
    assert telemetry["mode_effective"] == "manual"
    assert telemetry["profile_name"] == "calib_a"
    assert telemetry["profile_json"] == str(path)
    assert telemetry["profile_json_sha256"] == file_sha256(path)
    assert telemetry["profile_manifest_schema"] == HARDWARE_RESOLUTION_PROFILE_MANIFEST_SCHEMA
    assert telemetry["profile_schema"] == HARDWARE_RESOLUTION_GRADIENT_PROFILE_SCHEMA
    assert telemetry["profile_units"] == HARDWARE_RESOLUTION_PROFILE_UNITS
    assert telemetry["profile_provenance"]["source"] == "unit-test-explicit-calibration"


@pytest.mark.parametrize(
    "mode",
    ["ideal", "manual"],
)
def test_ideal_and_manual_reject_profile_source_arguments(tmp_path: Path, mode: str) -> None:
    path = _write_manifest(tmp_path, _profile_manifest())
    with pytest.raises(ValueError, match="rejects profile JSON/name"):
        resolve_hardware_resolution_config(
            mode=mode,
            gradient_hw_floor=0.0,
            gradient_drift_floor=0.0,
            profile_json=path,
            profile_name="calib_a",
        )


def test_ideal_uses_zero_floors_and_manual_uses_scalar_floors() -> None:
    ideal = resolve_hardware_resolution_config(
        mode="ideal",
        gradient_hw_floor=0.0,
        gradient_drift_floor=0.0,
    )
    manual = resolve_hardware_resolution_config(
        mode="manual",
        gradient_hw_floor=0.02,
        gradient_drift_floor=0.03,
    )

    assert ideal.effective_mode == "ideal"
    assert ideal.floor_source == "ideal_zero_floors"
    assert ideal.to_telemetry()["mode_effective"] == "ideal"
    assert manual.effective_mode == "manual"
    assert manual.gradient_hw_floor == pytest.approx(0.02)
    assert manual.gradient_drift_floor == pytest.approx(0.03)
    assert manual.floor_source == "manual_scalar_floors"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"profile_json": None, "profile_name": "calib_a"}, "profile_json is required"),
        ({"profile_json": "profiles.json", "profile_name": None}, "profile_name is required"),
        ({"profile_json": "profiles.json", "profile_name": ""}, "profile_name is required"),
    ],
)
def test_profile_mode_requires_json_and_profile_name(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_hardware_resolution_config(
            mode="profile",
            gradient_hw_floor=0.0,
            gradient_drift_floor=0.0,
            **kwargs,
        )


@pytest.mark.parametrize(
    "hw_floor,drift_floor",
    [(0.1, 0.0), (0.0, 0.1)],
)
def test_profile_mode_rejects_nonzero_scalar_cli_floors(
    tmp_path: Path,
    hw_floor: float,
    drift_floor: float,
) -> None:
    path = _write_manifest(tmp_path, _profile_manifest())
    with pytest.raises(ValueError, match="requires zero scalar"):
        resolve_hardware_resolution_config(
            mode="profile",
            gradient_hw_floor=hw_floor,
            gradient_drift_floor=drift_floor,
            profile_json=path,
            profile_name="calib_a",
        )


def test_profile_mode_rejects_bad_manifest_schema(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, _profile_manifest(manifest_schema="bad_schema"))
    with pytest.raises(ValueError, match="manifest schema"):
        resolve_hardware_resolution_config(
            mode="profile",
            gradient_hw_floor=0.0,
            gradient_drift_floor=0.0,
            profile_json=path,
            profile_name="calib_a",
        )


def test_profile_mode_rejects_missing_profiles_object(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        {"schema": HARDWARE_RESOLUTION_PROFILE_MANIFEST_SCHEMA, "profiles": []},
    )
    with pytest.raises(ValueError, match="profiles object"):
        resolve_hardware_resolution_config(
            mode="profile",
            gradient_hw_floor=0.0,
            gradient_drift_floor=0.0,
            profile_json=path,
            profile_name="calib_a",
        )


def test_profile_mode_rejects_missing_profile_name(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, _profile_manifest(name="calib_a"))
    with pytest.raises(ValueError, match="not found"):
        resolve_hardware_resolution_config(
            mode="profile",
            gradient_hw_floor=0.0,
            gradient_drift_floor=0.0,
            profile_json=path,
            profile_name="missing",
        )


@pytest.mark.parametrize(
    "payload,match",
    [
        (_profile_manifest(profile_schema="bad_profile_schema"), "gradient profile schema"),
        (_profile_manifest(profile_name="other"), "name mismatch"),
        (_profile_manifest(profile_name=None), "name mismatch"),
        (_profile_manifest(units="dimensionless"), "units"),
        (_profile_manifest(provenance="not-an-object"), "provenance"),
    ],
)
def test_profile_mode_rejects_bad_selected_profile_metadata(
    tmp_path: Path,
    payload: dict[str, object],
    match: str,
) -> None:
    path = _write_manifest(tmp_path, payload)
    with pytest.raises(ValueError, match=match):
        resolve_hardware_resolution_config(
            mode="profile",
            gradient_hw_floor=0.0,
            gradient_drift_floor=0.0,
            profile_json=path,
            profile_name="calib_a",
        )


@pytest.mark.parametrize(
    "bad_floor",
    [-0.001, float("nan"), float("inf"), "0.001", True],
)
def test_profile_mode_rejects_bad_json_floor_values(tmp_path: Path, bad_floor: object) -> None:
    path = _write_manifest(tmp_path, _profile_manifest(hw_floor=bad_floor))
    with pytest.raises(ValueError, match="JSON number|finite and nonnegative"):
        resolve_hardware_resolution_config(
            mode="profile",
            gradient_hw_floor=0.0,
            gradient_drift_floor=0.0,
            profile_json=path,
            profile_name="calib_a",
        )


def test_profile_mode_rejects_missing_json_floor(tmp_path: Path) -> None:
    manifest = _profile_manifest()
    del manifest["profiles"]["calib_a"]["gradient_drift_floor"]
    path = _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="missing required floor"):
        resolve_hardware_resolution_config(
            mode="profile",
            gradient_hw_floor=0.0,
            gradient_drift_floor=0.0,
            profile_json=path,
            profile_name="calib_a",
        )
