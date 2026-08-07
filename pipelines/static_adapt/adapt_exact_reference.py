"""Exact-reference manifest helpers for static ADAPT execution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


_EXACT_GS_REFERENCE_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "problem": ("problem", "family", "family_key"),
    "L": ("L", "num_sites", "sites"),
    "t": ("t", "J", "hopping"),
    "u": ("u", "U", "U_over_t"),
    "dv": ("dv", "delta_v"),
    "omega0": ("omega0", "omega", "phonon_frequency"),
    "g_ep": ("g_ep", "g", "electron_phonon_coupling"),
    "n_ph_max": ("n_ph_max", "n_ph_work", "working_cutoff"),
    "boson_encoding": ("boson_encoding", "encoding"),
    "ordering": ("ordering", "indexing"),
    "boundary": ("boundary",),
    "include_zero_point": ("include_zero_point",),
    "n_fermions": ("n_fermions", "num_particles"),
}


_EXACT_GS_REFERENCE_ENERGY_KEYS = (
    "exact_energy",
    "exact_gs_energy",
    "same_cutoff_exact_gs_energy",
    "exact_energy_filtered",
)


def _exact_reference_entries(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [entry for entry in payload if isinstance(entry, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("references", "entries", "rows", "exact_references"):
        raw = payload.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return [entry for entry in raw if isinstance(entry, Mapping)]
    return [payload]


def _exact_reference_entry_settings(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("match", "settings", "key"):
        value = entry.get(key)
        if isinstance(value, Mapping):
            return value
    return entry


def _nested_mapping_value(mapping: Mapping[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _exact_reference_entry_energy(entry: Mapping[str, Any]) -> float | None:
    for key in _EXACT_GS_REFERENCE_ENERGY_KEYS:
        raw = entry.get(key)
        if raw is not None:
            try:
                value = float(raw)
            except Exception:
                continue
            return value if math.isfinite(value) else None
    for path in (("ground_state", "exact_energy"), ("ground_state", "exact_energy_filtered"), ("adapt_vqe", "exact_gs_energy")):
        raw = _nested_mapping_value(entry, *path)
        if raw is not None:
            try:
                value = float(raw)
            except Exception:
                continue
            return value if math.isfinite(value) else None
    return None


def _lookup_exact_reference_field(settings: Mapping[str, Any], field: str) -> Any:
    for alias in _EXACT_GS_REFERENCE_FIELD_ALIASES.get(str(field), (str(field),)):
        if alias in settings:
            return settings.get(alias)
    return None


def _bool_from_any(value: Any) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _exact_reference_values_match(field: str, expected: Any, actual: Any) -> bool:
    if field in {"problem", "boson_encoding", "ordering", "boundary"}:
        return str(expected).strip().lower() == str(actual).strip().lower()
    if field in {"L", "n_ph_max", "n_fermions"}:
        if expected is None and actual in {None, ""}:
            return True
        try:
            return int(expected) == int(actual)
        except Exception:
            return False
    if field == "include_zero_point":
        actual_bool = _bool_from_any(actual)
        return actual_bool is not None and bool(expected) == bool(actual_bool)
    try:
        expected_f = float(expected)
        actual_f = float(actual)
    except Exception:
        return False
    return bool(math.isclose(expected_f, actual_f, rel_tol=1e-10, abs_tol=1e-10))


def _exact_reference_expected_key(args: argparse.Namespace) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "problem": str(args.problem),
        "L": int(args.L),
        "n_ph_max": int(args.n_ph_max),
        "boson_encoding": str(args.boson_encoding),
        "ordering": str(args.ordering),
        "boundary": str(args.boundary),
        "include_zero_point": bool(args.include_zero_point),
    }
    if getattr(args, "n_fermions", None) is not None:
        expected["n_fermions"] = int(args.n_fermions)
    if str(args.problem).strip().lower() == "hh":
        expected.update(
            {
                "t": float(args.t),
                "u": float(args.u),
                "dv": float(args.dv),
                "omega0": float(args.omega0),
                "g_ep": float(args.g_ep),
            }
        )
    return expected


def _resolve_exact_gs_energy_from_reference_json(
    path: Path,
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _exact_reference_entries(payload)
    expected = _exact_reference_expected_key(args)
    required_fields = tuple(expected.keys())
    near_misses: list[dict[str, Any]] = []
    for entry_idx, entry in enumerate(entries):
        settings = _exact_reference_entry_settings(entry)
        mismatches: list[str] = []
        for field in required_fields:
            actual = _lookup_exact_reference_field(settings, field)
            if actual is None:
                mismatches.append(f"missing:{field}")
                continue
            if not _exact_reference_values_match(field, expected[field], actual):
                mismatches.append(f"{field}: expected={expected[field]!r} actual={actual!r}")
        if mismatches:
            if len(near_misses) < 5:
                near_misses.append(
                    {
                        "entry_index": int(entry_idx),
                        "entry_id": entry.get("id", entry.get("regime", entry.get("case_id"))),
                        "mismatches": list(mismatches[:8]),
                    }
                )
            continue
        energy = _exact_reference_entry_energy(entry)
        if energy is None:
            raise ValueError(
                f"Exact-reference manifest entry {entry_idx} in {manifest_path} matches the run key but has no finite exact energy."
            )
        return float(energy), {
            "path": str(manifest_path),
            "entry_index": int(entry_idx),
            "entry_id": entry.get("id", entry.get("regime", entry.get("case_id"))),
            "matched_key": dict(expected),
            "source": str(entry.get("source", entry.get("exact_energy_source", "exact_reference_manifest"))),
        }
    raise ValueError(
        "No matching exact-reference entry found in "
        f"{manifest_path} for key {expected}. Near misses: {near_misses}"
    )
