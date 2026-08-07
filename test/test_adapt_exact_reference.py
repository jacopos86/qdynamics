from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipelines.static_adapt import adapt_exact_reference as exact_ref


def _args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "problem": "hh",
        "L": 2,
        "n_ph_max": 2,
        "boson_encoding": "binary",
        "ordering": "blocked",
        "boundary": "open",
        "include_zero_point": True,
        "n_fermions": None,
        "t": 1.0,
        "u": 0.25,
        "dv": 0.0,
        "omega0": 1.0,
        "g_ep": 0.3535533905932738,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_exact_reference_entries_and_nested_energy_helpers() -> None:
    rows = [{"id": "a"}, "skip", {"id": "b"}]
    assert exact_ref._exact_reference_entries(rows) == [{"id": "a"}, {"id": "b"}]
    assert exact_ref._exact_reference_entries({"references": rows}) == [{"id": "a"}, {"id": "b"}]
    assert exact_ref._exact_reference_entries("not-json-object") == []

    entry = {"settings": {"problem": "hh"}, "ground_state": {"exact_energy": "-1.25"}}
    assert exact_ref._exact_reference_entry_settings(entry) == {"problem": "hh"}
    assert exact_ref._nested_mapping_value(entry, "ground_state", "exact_energy") == "-1.25"
    assert exact_ref._exact_reference_entry_energy(entry) == pytest.approx(-1.25)
    assert exact_ref._exact_reference_entry_energy({"exact_energy": "nan"}) is None


def test_exact_reference_expected_key_and_value_alias_matching() -> None:
    expected = exact_ref._exact_reference_expected_key(_args(problem="HH", n_fermions=2))

    assert expected["problem"] == "HH"
    assert expected["L"] == 2
    assert expected["n_ph_max"] == 2
    assert expected["n_fermions"] == 2
    assert expected["g_ep"] == pytest.approx(0.3535533905932738)

    settings = {
        "family": "hh",
        "num_sites": "2",
        "working_cutoff": "2",
        "encoding": "BINARY",
        "indexing": "BLOCKED",
        "boundary": "OPEN",
        "include_zero_point": "yes",
        "U": "0.25",
    }
    assert exact_ref._lookup_exact_reference_field(settings, "problem") == "hh"
    assert exact_ref._lookup_exact_reference_field(settings, "L") == "2"
    assert exact_ref._lookup_exact_reference_field(settings, "n_ph_max") == "2"
    assert exact_ref._lookup_exact_reference_field(settings, "u") == "0.25"
    assert exact_ref._exact_reference_values_match("problem", "HH", "hh") is True
    assert exact_ref._exact_reference_values_match("include_zero_point", True, "yes") is True
    assert exact_ref._exact_reference_values_match("u", 0.25, "0.25000000001") is True
    assert exact_ref._exact_reference_values_match("L", 2, "bad") is False


def test_resolve_exact_gs_energy_from_reference_json_matches_alias_manifest(tmp_path: Path) -> None:
    manifest = _write_json(
        tmp_path / "exact_refs.json",
        {
            "references": [
                {
                    "id": "wrong-regime",
                    "match": {
                        "family": "hh",
                        "num_sites": 2,
                        "n_ph_work": 4,
                        "encoding": "binary",
                        "indexing": "blocked",
                        "boundary": "open",
                        "include_zero_point": True,
                        "J": 1.0,
                        "U": 0.25,
                        "delta_v": 0.0,
                        "omega": 1.0,
                        "g": 0.3535533905932738,
                    },
                    "exact_energy": -99.0,
                },
                {
                    "id": "weak_weak",
                    "match": {
                        "family": "HH",
                        "sites": "2",
                        "working_cutoff": "2",
                        "encoding": "BINARY",
                        "indexing": "BLOCKED",
                        "boundary": "OPEN",
                        "include_zero_point": "1",
                        "hopping": "1.0",
                        "U_over_t": "0.2500000000",
                        "dv": "0",
                        "phonon_frequency": "1.0",
                        "electron_phonon_coupling": "0.3535533905932738",
                    },
                    "ground_state": {"exact_energy_filtered": "-1.2345"},
                    "source": "tiny_fixture",
                },
            ]
        },
    )

    energy, meta = exact_ref._resolve_exact_gs_energy_from_reference_json(manifest, _args())

    assert energy == pytest.approx(-1.2345)
    assert meta["path"] == str(manifest)
    assert meta["entry_index"] == 1
    assert meta["entry_id"] == "weak_weak"
    assert meta["source"] == "tiny_fixture"
    assert meta["matched_key"]["problem"] == "hh"


def test_resolve_exact_gs_energy_reports_matching_entry_without_finite_energy(tmp_path: Path) -> None:
    manifest = _write_json(
        tmp_path / "exact_refs_bad_energy.json",
        {
            "rows": [
                {
                    "case_id": "match-no-energy",
                    "settings": exact_ref._exact_reference_expected_key(_args()),
                    "exact_energy": "nan",
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="matches the run key but has no finite exact energy"):
        exact_ref._resolve_exact_gs_energy_from_reference_json(manifest, _args())


def test_resolve_exact_gs_energy_reports_near_misses(tmp_path: Path) -> None:
    manifest = _write_json(
        tmp_path / "exact_refs_miss.json",
        {
            "entries": [
                {
                    "regime": "wrong-cutoff",
                    "key": {
                        **exact_ref._exact_reference_expected_key(_args()),
                        "n_ph_max": 4,
                    },
                    "exact_energy": -2.0,
                }
            ]
        },
    )

    with pytest.raises(ValueError) as excinfo:
        exact_ref._resolve_exact_gs_energy_from_reference_json(manifest, _args())

    message = str(excinfo.value)
    assert "No matching exact-reference entry found" in message
    assert "wrong-cutoff" in message
    assert "n_ph_max: expected=2 actual=4" in message


def test_adapt_pipeline_preserves_exact_reference_helper_import_compatibility() -> None:
    from pipelines.static_adapt import adapt_pipeline

    for name in (
        "_exact_reference_entries",
        "_exact_reference_entry_settings",
        "_nested_mapping_value",
        "_exact_reference_entry_energy",
        "_lookup_exact_reference_field",
        "_bool_from_any",
        "_exact_reference_values_match",
        "_exact_reference_expected_key",
        "_resolve_exact_gs_energy_from_reference_json",
    ):
        assert getattr(adapt_pipeline, name) is getattr(exact_ref, name)
