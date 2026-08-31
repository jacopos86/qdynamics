"""Contract for the content-addressed Paper-II exact-reference store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.time_dynamics.diagnostics import exact_reference_store as store_module
from pipelines.time_dynamics.diagnostics.exact_reference_store import (
    ExactReferenceStore,
)


def _run(path: Path, *, method: str, threshold: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "method": method,
                "threshold": threshold,
                "hamiltonian": {
                    "drive_profile": {
                        "A": 0.3,
                        "omega": 1.0,
                        "tbar": 1.0,
                        "phi": 0.0,
                        "t0": 0.0,
                        "pattern": "staggered",
                        "time_sampling": "midpoint",
                        "custom_weights": None,
                        "include_identity": False,
                        "operator_label": "drive",
                        "family_key": "hh",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _rows(times: list[float]) -> list[dict[str, object]]:
    return [
        {
            "time": time,
            "energy": -1.0 + time,
            "doublon": 0.1 + time,
            "site_occupations": [1.0 - time, 1.0 + time],
            "primary_density": 1.0 - time,
            "staggered": -2.0 * time,
            "statevector": np.asarray([1.0 + 0.0j, 0.0 + 0.0j]),
        }
        for time in times
    ]


def test_method_and_threshold_do_not_change_physical_reference_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = tmp_path / "seed.json"
    seed.write_text("seed", encoding="utf-8")
    apm = _run(tmp_path / "apm.json", method="apm", threshold=1.0e-3)
    avqds = _run(tmp_path / "avqds.json", method="avqds", threshold=1.0e-6)
    calls: list[Path] = []

    def fake_exact(seed_path, run_path, *, times, substeps):
        calls.append(Path(run_path))
        return _rows(list(times))

    monkeypatch.setattr(store_module, "exact_rows_on_times", fake_exact)
    store = ExactReferenceStore(tmp_path / "references")
    first = store.get_or_build(
        seed_path=seed, run_path=apm, times=[0.0, 0.5], substeps=40
    )
    second = store.get_or_build(
        seed_path=seed, run_path=avqds, times=[0.0, 0.5], substeps=40
    )

    assert first.key == second.key
    assert calls == [apm]
    assert [row["energy"] for row in second.rows] == [-1.0, -0.5]


def test_verified_read_rejects_modified_reference_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = tmp_path / "seed.json"
    seed.write_text("seed", encoding="utf-8")
    run = _run(tmp_path / "run.json", method="apm", threshold=1.0e-3)
    monkeypatch.setattr(
        store_module,
        "exact_rows_on_times",
        lambda *args, times, **kwargs: _rows(list(times)),
    )
    store = ExactReferenceStore(tmp_path / "references")
    artifact = store.get_or_build(
        seed_path=seed, run_path=run, times=[0.0, 1.0], substeps=40
    )
    artifact.data_path.write_bytes(artifact.data_path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="digest mismatch"):
        store.get_or_build(
            seed_path=seed, run_path=run, times=[0.0, 1.0], substeps=40
        )
