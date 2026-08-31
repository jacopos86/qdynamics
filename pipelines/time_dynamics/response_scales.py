"""Verified exact-response scales keyed by complete physical cells.

Historical Paper-II ledgers embed the original ``drive -> horizon`` mapping.
That representation remains replayable, but it cannot distinguish a drive at
different phonon cutoffs, interaction regimes, or controller cells.  New
campaign construction normalizes a verified sidecar into the v2 index here:

``regime -> controller -> drive -> horizon -> scale row``.

Every row also carries its explicit physical identity.  The exact trajectory
is controller-independent, but the controller remains part of the comparison
cell identity so a scale cannot be silently reused outside its registered
matrix cell.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

RESPONSE_SCALE_INDEX_SCHEMA_V2 = "paper_ii_response_scale_index_v2"
_SIDECAR_SCHEMA_V1 = "paper_ii_exact_response_scales_v1"
_REFERENCE_SCHEMA_V1 = "paper_ii_exact_reference_store_v1"
_HORIZON_FINAL_TIME = {"t5": 5.0, "t20": 20.0}
_SCALE_FIELDS = (
    "energy_rms_excursion",
    "doublon_rms_excursion",
    "max_site_occupation_rms_excursion",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PHYSICAL_IDENTITY_FIELDS = (
    "model",
    "family_key",
    "regime",
    "seed_sha256",
    "lattice_sites",
    "n_ph_max",
    "boundary",
    "boson_encoding",
    "orbital_ordering",
    "hopping",
    "next_nearest_hopping",
    "staggered_potential",
    "phonon_frequency",
    "U",
    "g",
    "nearest_neighbor_interaction",
    "n_fermions",
    "initial_state_source",
)


class ResponseScaleIntegrityError(ValueError):
    """The sidecar or one of its exact-reference bindings is unverifiable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ResponseScaleIntegrityError(
            f"cannot hash response-scale provenance file {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResponseScaleIntegrityError(
            f"{label} is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ResponseScaleIntegrityError(f"{label} is not a JSON object: {path}")
    return dict(payload)


def _relative_or_absolute(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _finite_positive(value: Any, *, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ResponseScaleIntegrityError(f"{label} is not numeric") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ResponseScaleIntegrityError(f"{label} must be finite and positive")
    return numeric


def _finite_number(value: Any, *, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ResponseScaleIntegrityError(f"{label} is not numeric") from exc
    if not math.isfinite(numeric):
        raise ResponseScaleIntegrityError(f"{label} must be finite")
    return numeric


def _sha256_value(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if _SHA256.fullmatch(digest) is None:
        raise ResponseScaleIntegrityError(f"{label} is not a lowercase SHA-256")
    return digest


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _physical_identity(
    *,
    seed: Mapping[str, Any],
    seed_sha256: str,
    regime_id: str,
    family_key: str,
) -> dict[str, Any]:
    settings = seed.get("settings")
    if not isinstance(settings, Mapping):
        raise ResponseScaleIntegrityError("seed artifact has no settings object")
    required = (
        "L",
        "n_ph_max",
        "boundary",
        "boson_encoding",
        "ordering",
        "t",
        "t_prime",
        "dv",
        "omega0",
        "u",
        "g_ep",
        "v_nn",
        "initial_state_source",
        "problem",
    )
    missing = [source for source in required if settings.get(source) is None]
    if missing:
        raise ResponseScaleIntegrityError(
            f"seed settings omit physical identity fields {missing}"
        )
    return {
        "model": str(settings["problem"]),
        "family_key": str(family_key),
        "regime": str(regime_id),
        "seed_sha256": _sha256_value(seed_sha256, label="seed SHA-256"),
        "lattice_sites": int(settings["L"]),
        "n_ph_max": int(settings["n_ph_max"]),
        "boundary": str(settings["boundary"]),
        "boson_encoding": str(settings["boson_encoding"]),
        "orbital_ordering": str(settings["ordering"]),
        "hopping": _finite_number(settings["t"], label="seed hopping"),
        "next_nearest_hopping": _finite_number(
            settings["t_prime"], label="seed next-nearest hopping"
        ),
        "staggered_potential": _finite_number(
            settings["dv"], label="seed staggered potential"
        ),
        "phonon_frequency": _finite_number(
            settings["omega0"], label="seed phonon frequency"
        ),
        "U": _finite_number(settings["u"], label="seed Hubbard U"),
        "g": _finite_number(settings["g_ep"], label="seed electron-phonon g"),
        "nearest_neighbor_interaction": _finite_number(
            settings["v_nn"], label="seed nearest-neighbor interaction"
        ),
        "n_fermions": (
            None
            if settings.get("n_fermions") is None
            else int(settings["n_fermions"])
        ),
        "initial_state_source": str(settings["initial_state_source"]),
    }


def _drive_identity(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return every exact-reference drive field that fixes the trajectory."""

    spatial_weights = profile.get("spatial_weights")
    if not isinstance(spatial_weights, Sequence) or isinstance(
        spatial_weights, (str, bytes)
    ):
        raise ResponseScaleIntegrityError("exact-reference spatial weights are invalid")
    custom_weights = profile.get("custom_weights")
    if custom_weights is not None and (
        not isinstance(custom_weights, Sequence)
        or isinstance(custom_weights, (str, bytes))
    ):
        raise ResponseScaleIntegrityError("exact-reference custom weights are invalid")
    identity = {
        "family_key": str(profile.get("family_key") or ""),
        "operator_label": str(profile.get("operator_label") or ""),
        "pattern": str(profile.get("pattern") or ""),
        "amplitude": _finite_number(
            profile.get("A", profile.get("amplitude")), label="drive amplitude"
        ),
        "omega": _finite_number(profile.get("omega"), label="drive omega"),
        "phase": _finite_number(
            profile.get("phi", profile.get("phase")), label="drive phase"
        ),
        "start_time": _finite_number(
            profile.get("t0", profile.get("start_time")),
            label="drive start time",
        ),
        "time_scale": _finite_number(
            profile.get("tbar", profile.get("time_scale")),
            label="drive time scale",
        ),
        "include_identity": bool(profile.get("include_identity")),
        "spatial_weights": [
            _finite_number(value, label="drive spatial weight")
            for value in spatial_weights
        ],
        "custom_weights": (
            None
            if custom_weights is None
            else [
                _finite_number(value, label="drive custom weight")
                for value in custom_weights
            ]
        ),
        "time_sampling": str(profile.get("time_sampling") or ""),
        "reference_method_name": str(
            profile.get("reference_method_name") or ""
        ),
    }
    for field in (
        "family_key",
        "operator_label",
        "pattern",
        "time_sampling",
        "reference_method_name",
    ):
        if not identity[field]:
            raise ResponseScaleIntegrityError(
                f"exact-reference drive profile omits {field}"
            )
    return identity


def _verified_reference(
    *,
    exact_reference_root: Path,
    reference_key: str,
    expected_manifest_sha256: str,
    expected_seed_sha256: str,
    expected_model: str,
    expected_amplitude: float,
    expected_omega: float,
) -> tuple[dict[str, Any], Mapping[str, np.ndarray], dict[str, Any]]:
    reference_dir = (exact_reference_root / reference_key).resolve()
    try:
        reference_dir.relative_to(exact_reference_root.resolve())
    except ValueError as exc:
        raise ResponseScaleIntegrityError(
            f"exact-reference key escapes its store: {reference_key!r}"
        ) from exc
    manifest_path = reference_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ResponseScaleIntegrityError(
            f"exact-reference manifest is missing: {manifest_path}"
        )
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise ResponseScaleIntegrityError(
            "exact-reference manifest SHA-256 mismatch for "
            f"{reference_key}: expected {expected_manifest_sha256}, "
            f"got {manifest_sha256}"
        )
    manifest = _read_json(manifest_path, label="exact-reference manifest")
    if manifest.get("schema") != _REFERENCE_SCHEMA_V1:
        raise ResponseScaleIntegrityError(
            f"unsupported exact-reference schema for {reference_key}"
        )
    if manifest.get("key") != reference_key:
        raise ResponseScaleIntegrityError(
            f"exact-reference key mismatch for {reference_key}"
        )
    if manifest.get("seed_sha256") != expected_seed_sha256:
        raise ResponseScaleIntegrityError(
            f"exact-reference seed mismatch for {reference_key}"
        )
    profile = manifest.get("drive_profile")
    if not isinstance(profile, Mapping):
        raise ResponseScaleIntegrityError(
            f"exact-reference drive profile is missing for {reference_key}"
        )
    if str(profile.get("family_key") or "") != expected_model:
        raise ResponseScaleIntegrityError(
            f"exact-reference model mismatch for {reference_key}"
        )
    drive_identity = _drive_identity(profile)
    if not math.isclose(
        float(drive_identity["amplitude"]),
        expected_amplitude,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    ) or not math.isclose(
        float(drive_identity["omega"]),
        expected_omega,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    ):
        raise ResponseScaleIntegrityError(
            f"exact-reference drive mismatch for {reference_key}"
        )
    data_name = Path(str(manifest.get("data_file") or ""))
    if not data_name.parts or data_name.is_absolute() or ".." in data_name.parts:
        raise ResponseScaleIntegrityError(
            f"unsafe exact-reference data path for {reference_key}"
        )
    data_path = (reference_dir / data_name).resolve()
    try:
        data_path.relative_to(reference_dir)
    except ValueError as exc:
        raise ResponseScaleIntegrityError(
            f"exact-reference data escapes its store for {reference_key}"
        ) from exc
    if not data_path.is_file() or _sha256(data_path) != manifest.get("data_sha256"):
        raise ResponseScaleIntegrityError(
            f"exact-reference data SHA-256 mismatch for {reference_key}"
        )
    try:
        loaded = np.load(data_path, allow_pickle=False)
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        loaded.close()
    except (OSError, ValueError, KeyError) as exc:
        raise ResponseScaleIntegrityError(
            f"exact-reference data is unreadable for {reference_key}: {exc}"
        ) from exc
    required_arrays = {"time", "energy", "doublon", "site_occupations"}
    if not required_arrays.issubset(arrays):
        raise ResponseScaleIntegrityError(
            f"exact-reference data omits reporting arrays for {reference_key}"
        )
    times = np.asarray(arrays["time"], dtype=float)
    manifest_times = np.asarray(manifest.get("times") or (), dtype=float)
    if (
        times.ndim != 1
        or int(manifest.get("row_count", -1)) != len(times)
        or manifest_times.shape != times.shape
        or not np.array_equal(manifest_times, times)
        or len(times) < 2
        or not math.isclose(float(times[0]), 0.0, abs_tol=1.0e-14)
        or not math.isclose(float(times[-1]), 20.0, abs_tol=1.0e-12)
        or not np.allclose(np.diff(times), 0.04, rtol=0.0, atol=1.0e-12)
    ):
        raise ResponseScaleIntegrityError(
            f"exact-reference grid mismatch for {reference_key}"
        )
    return manifest, arrays, drive_identity


def _calculated_scales(
    arrays: Mapping[str, np.ndarray], *, final_time: float
) -> dict[str, float]:
    times = np.asarray(arrays["time"], dtype=float)
    energy_all = np.asarray(arrays["energy"], dtype=float)
    doublon_all = np.asarray(arrays["doublon"], dtype=float)
    occupations_all = np.asarray(arrays["site_occupations"], dtype=float)
    if (
        times.ndim != 1
        or energy_all.shape != times.shape
        or doublon_all.shape != times.shape
        or occupations_all.ndim != 2
        or occupations_all.shape[0] != len(times)
        or not np.all(np.isfinite(times))
        or not np.all(np.isfinite(energy_all))
        or not np.all(np.isfinite(doublon_all))
        or not np.all(np.isfinite(occupations_all))
    ):
        raise ResponseScaleIntegrityError(
            "exact-reference observable shapes or values are invalid"
        )
    mask = times <= float(final_time) + 1.0e-12
    energy = energy_all[mask]
    doublon = doublon_all[mask]
    occupations = occupations_all[mask]
    expected_rows = int(round(float(final_time) / 0.04)) + 1
    if int(np.count_nonzero(mask)) != expected_rows:
        raise ResponseScaleIntegrityError(
            f"exact-reference horizon {final_time:g} has the wrong row count"
        )
    return {
        "energy_rms_excursion": float(
            np.sqrt(np.mean(np.square(energy - energy[0])))
        ),
        "doublon_rms_excursion": float(
            np.sqrt(np.mean(np.square(doublon - doublon[0])))
        ),
        "max_site_occupation_rms_excursion": float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.max(np.abs(occupations - occupations[0]), axis=1)
                    )
                )
            )
        ),
    }


def load_verified_response_scale_index(
    sidecar_path: str | Path,
    *,
    expected_sidecar_sha256: str,
    exact_reference_root: str | Path,
    seed_path: str | Path,
    regime_id: str,
    family_key: str,
    controller_ids: Sequence[str],
    expected_drives: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Load the legacy baseline sidecar into a verified v2 cell index."""

    sidecar = Path(sidecar_path)
    actual_sidecar_sha256 = _sha256(sidecar)
    if actual_sidecar_sha256 != expected_sidecar_sha256:
        raise ResponseScaleIntegrityError(
            "response-scale sidecar SHA-256 mismatch: expected "
            f"{expected_sidecar_sha256}, got {actual_sidecar_sha256}"
        )
    payload = _read_json(sidecar, label="response-scale sidecar")
    if payload.get("schema") != _SIDECAR_SCHEMA_V1:
        raise ResponseScaleIntegrityError("unsupported response-scale sidecar schema")
    drives = payload.get("drives")
    if not isinstance(drives, Mapping) or set(drives) != set(expected_drives):
        raise ResponseScaleIntegrityError(
            "response-scale sidecar drive identity does not match the campaign"
        )
    controllers = tuple(str(value) for value in controller_ids)
    if (
        not controllers
        or any(not value.strip() for value in controllers)
        or len(set(controllers)) != len(controllers)
    ):
        raise ResponseScaleIntegrityError(
            "controller identities must be unique and nonempty"
        )

    seed = Path(seed_path)
    if not seed.is_file():
        raise ResponseScaleIntegrityError(f"response-scale seed is missing: {seed}")
    seed_sha256 = _sha256(seed)
    seed_payload = _read_json(seed, label="response-scale seed")
    physical = _physical_identity(
        seed=seed_payload,
        seed_sha256=seed_sha256,
        regime_id=regime_id,
        family_key=family_key,
    )
    reference_root = Path(exact_reference_root)
    repository_root = reference_root.resolve().parents[2]
    verified_drives: dict[str, dict[str, dict[str, Any]]] = {}
    reference_cache: dict[
        tuple[str, str, str, str, float, float],
        tuple[dict[str, Any], Mapping[str, np.ndarray], dict[str, Any]],
    ] = {}
    for drive_id in sorted(expected_drives):
        horizons = drives[drive_id]
        if not isinstance(horizons, Mapping) or set(horizons) != set(
            _HORIZON_FINAL_TIME
        ):
            raise ResponseScaleIntegrityError(
                f"response scales for {drive_id!r} must bind t5 and t20"
            )
        expected_drive = expected_drives[drive_id]
        amplitude = _finite_number(
            expected_drive.get("amplitude"), label=f"{drive_id} amplitude"
        )
        omega = _finite_positive(
            expected_drive.get("omega"), label=f"{drive_id} omega"
        )
        verified_horizons: dict[str, dict[str, Any]] = {}
        for horizon, final_time in _HORIZON_FINAL_TIME.items():
            raw_row = horizons[horizon]
            if not isinstance(raw_row, Mapping):
                raise ResponseScaleIntegrityError(
                    f"response-scale row is invalid for {drive_id}/{horizon}"
                )
            reference_key = str(raw_row.get("exact_reference_key") or "")
            manifest_sha256 = str(
                raw_row.get("exact_reference_manifest_sha256") or ""
            )
            if not reference_key:
                raise ResponseScaleIntegrityError(
                    f"response-scale provenance is incomplete for {drive_id}/{horizon}"
                )
            manifest_sha256 = _sha256_value(
                manifest_sha256,
                label=f"{drive_id}/{horizon} exact-reference manifest SHA-256",
            )
            cache_key = (
                reference_key,
                manifest_sha256,
                seed_sha256,
                str(physical["model"]),
                amplitude,
                omega,
            )
            if cache_key not in reference_cache:
                reference_cache[cache_key] = _verified_reference(
                    exact_reference_root=reference_root,
                    reference_key=reference_key,
                    expected_manifest_sha256=manifest_sha256,
                    expected_seed_sha256=seed_sha256,
                    expected_model=str(physical["model"]),
                    expected_amplitude=amplitude,
                    expected_omega=omega,
                )
            manifest, arrays, drive_identity = reference_cache[cache_key]
            manifest_path = reference_root / reference_key / "manifest.json"
            if _sha256(manifest_path) != manifest_sha256:
                raise ResponseScaleIntegrityError(
                    f"exact-reference manifest SHA-256 mismatch for {reference_key}"
                )
            calculated = _calculated_scales(arrays, final_time=final_time)
            row: dict[str, Any] = {
                name: _finite_positive(
                    raw_row.get(name), label=f"{drive_id}/{horizon}/{name}"
                )
                for name in _SCALE_FIELDS
            }
            for name, value in calculated.items():
                if not math.isclose(
                    row[name], value, rel_tol=1.0e-13, abs_tol=1.0e-15
                ):
                    raise ResponseScaleIntegrityError(
                        "response scale disagrees with exact data for "
                        f"{drive_id}/{horizon}/{name}"
                    )
            row.update(
                {
                    "exact_reference_key": reference_key,
                    "exact_reference_manifest_sha256": manifest_sha256,
                    "exact_reference_data_sha256": _sha256_value(
                        manifest.get("data_sha256"),
                        label=f"{reference_key} data SHA-256",
                    ),
                    "drive_identity": drive_identity,
                }
            )
            verified_horizons[horizon] = row
        if (
            verified_horizons["t5"]["exact_reference_key"]
            != verified_horizons["t20"]["exact_reference_key"]
        ):
            raise ResponseScaleIntegrityError(
                f"t5/t20 exact-reference identity differs for {drive_id}"
            )
        verified_drives[drive_id] = verified_horizons

    controller_rows: dict[str, Any] = {}
    for controller in controllers:
        drive_rows: dict[str, Any] = {}
        for drive_id, horizons in verified_drives.items():
            horizon_rows: dict[str, Any] = {}
            for horizon, scales in horizons.items():
                final_time = _HORIZON_FINAL_TIME[horizon]
                sample_count = int(round(final_time / 0.04)) + 1
                identity = {
                    **physical,
                    "controller": controller,
                    "drive": drive_id,
                    "drive_amplitude": float(expected_drives[drive_id]["amplitude"]),
                    "drive_omega": float(expected_drives[drive_id]["omega"]),
                    "drive_profile": dict(scales["drive_identity"]),
                    "horizon": horizon,
                    "horizon_final_time": final_time,
                    "horizon_sample_count": sample_count,
                    "horizon_time_step": 0.04,
                }
                row = dict(scales)
                row.pop("drive_identity")
                horizon_rows[horizon] = {
                    **row,
                    "cell_identity": identity,
                    "cell_identity_sha256": _canonical_json_sha256(identity),
                }
            drive_rows[drive_id] = horizon_rows
        controller_rows[controller] = {"drives": drive_rows}

    index = {
        "schema": RESPONSE_SCALE_INDEX_SCHEMA_V2,
        "source": {
            "path": _relative_or_absolute(sidecar, repository_root),
            "sha256": actual_sidecar_sha256,
            "schema": _SIDECAR_SCHEMA_V1,
            "definition": str(payload.get("definition") or ""),
            "seed_path": _relative_or_absolute(seed, repository_root),
            "seed_sha256": seed_sha256,
            "exact_reference_root": _relative_or_absolute(
                reference_root, repository_root
            ),
        },
        "regimes": {
            str(regime_id): {
                "physical_identity": physical,
                "controllers": controller_rows,
            }
        },
    }
    validate_response_scale_index(index)
    return index


def is_response_scale_index(payload: Mapping[str, Any]) -> bool:
    return payload.get("schema") == RESPONSE_SCALE_INDEX_SCHEMA_V2


def validate_response_scale_index(payload: Mapping[str, Any]) -> None:
    if not is_response_scale_index(payload):
        raise ResponseScaleIntegrityError("unsupported response-scale index schema")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ResponseScaleIntegrityError("response-scale index has no source binding")
    _sha256_value(source.get("sha256"), label="response-scale source SHA-256")
    _sha256_value(source.get("seed_sha256"), label="response-scale seed SHA-256")
    regimes = payload.get("regimes")
    if not isinstance(regimes, Mapping) or not regimes:
        raise ResponseScaleIntegrityError("response-scale index has no regimes")
    for regime, regime_row in regimes.items():
        if not isinstance(regime_row, Mapping):
            raise ResponseScaleIntegrityError(
                f"invalid response-scale regime {regime!r}"
            )
        physical = regime_row.get("physical_identity")
        if not isinstance(physical, Mapping) or physical.get("regime") != regime:
            raise ResponseScaleIntegrityError(
                f"response-scale physical identity mismatch for {regime!r}"
            )
        if set(physical) != set(_PHYSICAL_IDENTITY_FIELDS):
            raise ResponseScaleIntegrityError(
                f"response-scale regime {regime!r} has incomplete physical identity"
            )
        for required in (
            "model",
            "family_key",
            "boundary",
            "boson_encoding",
            "orbital_ordering",
            "initial_state_source",
        ):
            if not str(physical.get(required) or ""):
                raise ResponseScaleIntegrityError(
                    f"response-scale regime {regime!r} omits {required}"
                )
        _sha256_value(
            physical.get("seed_sha256"), label=f"{regime} physical seed SHA-256"
        )
        if physical.get("seed_sha256") != source.get("seed_sha256"):
            raise ResponseScaleIntegrityError(
                f"response-scale regime {regime!r} seed binding differs from source"
            )
        for required in ("lattice_sites", "n_ph_max"):
            try:
                numeric = int(physical[required])
            except (TypeError, ValueError) as exc:
                raise ResponseScaleIntegrityError(
                    f"response-scale regime {regime!r} has invalid {required}"
                ) from exc
            if numeric <= 0 or numeric != physical[required]:
                raise ResponseScaleIntegrityError(
                    f"response-scale regime {regime!r} has invalid {required}"
                )
        for required in (
            "hopping",
            "next_nearest_hopping",
            "staggered_potential",
            "phonon_frequency",
            "U",
            "g",
            "nearest_neighbor_interaction",
        ):
            _finite_number(physical.get(required), label=f"{regime}/{required}")
        controllers = regime_row.get("controllers")
        if not isinstance(controllers, Mapping) or not controllers:
            raise ResponseScaleIntegrityError(
                f"response-scale regime {regime!r} has no controllers"
            )
        for controller, controller_row in controllers.items():
            drives = (
                controller_row.get("drives")
                if isinstance(controller_row, Mapping)
                else None
            )
            if not isinstance(drives, Mapping) or not drives:
                raise ResponseScaleIntegrityError(
                    f"response-scale controller {controller!r} has no drives"
                )
            for drive, horizons in drives.items():
                if not isinstance(horizons, Mapping) or not horizons:
                    raise ResponseScaleIntegrityError(
                        f"response-scale drive {drive!r} has no horizons"
                    )
                for horizon, row in horizons.items():
                    if not isinstance(row, Mapping):
                        raise ResponseScaleIntegrityError(
                            f"invalid response-scale row for {drive}/{horizon}"
                        )
                    identity = row.get("cell_identity")
                    if not isinstance(identity, Mapping):
                        raise ResponseScaleIntegrityError(
                            "response-scale row has no identity for "
                            f"{regime}/{controller}/{drive}/{horizon}"
                        )
                    drive_profile = identity.get("drive_profile")
                    if not isinstance(drive_profile, Mapping):
                        raise ResponseScaleIntegrityError(
                            "response-scale drive identity is missing for "
                            f"{regime}/{controller}/{drive}/{horizon}"
                        )
                    normalized_drive = _drive_identity(drive_profile)
                    amplitude = _finite_number(
                        identity.get("drive_amplitude"), label="drive amplitude"
                    )
                    omega = _finite_positive(
                        identity.get("drive_omega"), label="drive omega"
                    )
                    if not math.isclose(
                        amplitude,
                        float(normalized_drive["amplitude"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-14,
                    ) or not math.isclose(
                        omega,
                        float(normalized_drive["omega"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-14,
                    ):
                        raise ResponseScaleIntegrityError(
                            "response-scale drive coordinates disagree for "
                            f"{regime}/{controller}/{drive}/{horizon}"
                        )
                    if horizon not in _HORIZON_FINAL_TIME:
                        raise ResponseScaleIntegrityError(
                            f"unsupported response-scale horizon {horizon!r}"
                        )
                    final_time = _HORIZON_FINAL_TIME[horizon]
                    sample_count = int(round(final_time / 0.04)) + 1
                    expected = {
                        **dict(physical),
                        "controller": controller,
                        "drive": drive,
                        "drive_amplitude": amplitude,
                        "drive_omega": omega,
                        "drive_profile": normalized_drive,
                        "horizon": horizon,
                        "horizon_final_time": final_time,
                        "horizon_sample_count": sample_count,
                        "horizon_time_step": 0.04,
                    }
                    if dict(identity) != expected:
                        raise ResponseScaleIntegrityError(
                            "response-scale cell identity mismatch for "
                            f"{regime}/{controller}/{drive}/{horizon}"
                        )
                    if row.get("cell_identity_sha256") != _canonical_json_sha256(
                        expected
                    ):
                        raise ResponseScaleIntegrityError(
                            "response-scale cell identity digest mismatch for "
                            f"{regime}/{controller}/{drive}/{horizon}"
                        )
                    for name in _SCALE_FIELDS:
                        _finite_positive(row.get(name), label=name)
                    for name in (
                        "exact_reference_manifest_sha256",
                        "exact_reference_data_sha256",
                    ):
                        _sha256_value(
                            row.get(name),
                            label=(
                                f"{regime}/{controller}/{drive}/{horizon}/{name}"
                            ),
                        )


def response_scale_row(
    payload: Mapping[str, Any],
    *,
    regime: str,
    controller: str,
    drive: str,
    horizon: str,
) -> Mapping[str, Any]:
    try:
        row = payload["regimes"][regime]["controllers"][controller]["drives"][
            drive
        ][horizon]
    except (KeyError, TypeError) as exc:
        raise ResponseScaleIntegrityError(
            "no response scale is bound for "
            f"{regime}/{controller}/{drive}/{horizon}"
        ) from exc
    if not isinstance(row, Mapping):
        raise ResponseScaleIntegrityError(
            f"invalid response scale for {regime}/{controller}/{drive}/{horizon}"
        )
    return row


def response_scale_cell_identity(
    payload: Mapping[str, Any],
    *,
    regime: str,
    controller: str,
    drive: str,
    horizon: str,
) -> dict[str, Any]:
    row = response_scale_row(
        payload,
        regime=regime,
        controller=controller,
        drive=drive,
        horizon=horizon,
    )
    identity = row.get("cell_identity")
    if not isinstance(identity, Mapping):
        raise ResponseScaleIntegrityError("response-scale row has no cell identity")
    return dict(identity)


__all__ = [
    "RESPONSE_SCALE_INDEX_SCHEMA_V2",
    "ResponseScaleIntegrityError",
    "is_response_scale_index",
    "load_verified_response_scale_index",
    "response_scale_cell_identity",
    "response_scale_row",
    "validate_response_scale_index",
]
