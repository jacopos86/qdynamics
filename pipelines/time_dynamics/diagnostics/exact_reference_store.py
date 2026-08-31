"""Content-addressed exact references for Paper-II trajectory diagnostics.

One physical seed, drive profile, reporting grid, propagator schema, and
substep count identify one immutable reference.  Method, structural policy,
time-step controller, and activation threshold are deliberately absent from
the key: those runs must all be compared with the same physical trajectory.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.diagnostics.exact_driven_reference import (
    exact_rows_on_times,
)


SCHEMA = "paper_ii_exact_reference_store_v1"
PROPAGATOR_SCHEMA = "exponential_midpoint_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class ExactReferenceArtifact:
    """One verified immutable exact-reference artifact."""

    key: str
    manifest_path: Path
    data_path: Path
    rows: tuple[Mapping[str, Any], ...]

    @property
    def data_sha256(self) -> str:
        return _sha256(self.data_path)


class ExactReferenceStore:
    """Build once and verify on every read; never silently replace a reference."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def get_or_build(
        self,
        *,
        seed_path: str | Path,
        run_path: str | Path,
        times: Sequence[float],
        substeps: int = 40,
    ) -> ExactReferenceArtifact:
        seed = Path(seed_path)
        run = Path(run_path)
        time_grid = [float(value) for value in times]
        run_doc = json.loads(run.read_text(encoding="utf-8"))
        identity = {
            "schema": SCHEMA,
            "propagator_schema": PROPAGATOR_SCHEMA,
            "seed_sha256": _sha256(seed),
            "drive_profile": run_doc["hamiltonian"]["drive_profile"],
            "times": time_grid,
            "substeps": int(substeps),
        }
        key = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        artifact_root = self.root / key
        manifest_path = artifact_root / "manifest.json"
        data_path = artifact_root / "reference.npz"
        if manifest_path.exists() or data_path.exists():
            return self._load_verified(
                key=key,
                identity=identity,
                manifest_path=manifest_path,
                data_path=data_path,
            )

        rows = exact_rows_on_times(
            seed,
            run,
            times=time_grid,
            substeps=int(substeps),
        )
        arrays = self._rows_to_arrays(rows)
        artifact_root.mkdir(parents=True, exist_ok=False)
        temporary_data = artifact_root / "reference.npz.part"
        with temporary_data.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_data, data_path)
        manifest = {
            **identity,
            "key": key,
            "row_count": len(rows),
            "data_file": data_path.name,
            "data_sha256": _sha256(data_path),
        }
        temporary_manifest = artifact_root / "manifest.json.part"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
        return self._load_verified(
            key=key,
            identity=identity,
            manifest_path=manifest_path,
            data_path=data_path,
        )

    @staticmethod
    def _rows_to_arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
        return {
            "time": np.asarray([row["time"] for row in rows], dtype=float),
            "energy": np.asarray([row["energy"] for row in rows], dtype=float),
            "doublon": np.asarray([row["doublon"] for row in rows], dtype=float),
            "site_occupations": np.asarray(
                [row["site_occupations"] for row in rows], dtype=float
            ),
            "primary_density": np.asarray(
                [row["primary_density"] for row in rows], dtype=float
            ),
            "staggered": np.asarray([row["staggered"] for row in rows], dtype=float),
            "state_real": np.asarray(
                [np.asarray(row["statevector"]).real for row in rows], dtype=float
            ),
            "state_imag": np.asarray(
                [np.asarray(row["statevector"]).imag for row in rows], dtype=float
            ),
        }

    def _load_verified(
        self,
        *,
        key: str,
        identity: Mapping[str, Any],
        manifest_path: Path,
        data_path: Path,
    ) -> ExactReferenceArtifact:
        if not manifest_path.is_file() or not data_path.is_file():
            raise ValueError(f"incomplete exact-reference artifact: {key}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field, expected in identity.items():
            if manifest.get(field) != expected:
                raise ValueError(f"exact-reference manifest mismatch for {field}")
        if manifest.get("key") != key:
            raise ValueError("exact-reference content key mismatch")
        if manifest.get("data_sha256") != _sha256(data_path):
            raise ValueError("exact-reference data digest mismatch")
        with np.load(data_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        required = {
            "time",
            "energy",
            "doublon",
            "site_occupations",
            "primary_density",
            "staggered",
            "state_real",
            "state_imag",
        }
        if set(arrays) != required:
            raise ValueError("exact-reference array schema mismatch")
        count = int(manifest["row_count"])
        if any(array.shape[0] != count for array in arrays.values()):
            raise ValueError("exact-reference row count mismatch")
        if not np.array_equal(arrays["time"], np.asarray(identity["times"], dtype=float)):
            raise ValueError("exact-reference reporting grid mismatch")
        rows = tuple(
            {
                "time": float(arrays["time"][index]),
                "energy": float(arrays["energy"][index]),
                "doublon": float(arrays["doublon"][index]),
                "site_occupations": arrays["site_occupations"][index].tolist(),
                "primary_density": float(arrays["primary_density"][index]),
                "staggered": float(arrays["staggered"][index]),
                "statevector": arrays["state_real"][index]
                + 1j * arrays["state_imag"][index],
            }
            for index in range(count)
        )
        return ExactReferenceArtifact(
            key=key,
            manifest_path=manifest_path,
            data_path=data_path,
            rows=rows,
        )


__all__ = ["ExactReferenceArtifact", "ExactReferenceStore"]
