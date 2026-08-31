"""Immutable, reporting-only exact references for finite-box SIAM runs.

The reference key deliberately excludes the variational method, activation
threshold, and controller.  It binds only the verified prepared-state seed,
the SIAM physics encoded by that seed, the complete ordered reporting grid,
and the exact propagator schema.  Consequently every comparable structural
arm is scored against the same bytes.

This module has no dependency on an online trajectory controller.  It is an
offline reporting boundary and exact data produced here must never be passed
back into propagation or structural selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from pipelines.time_dynamics.siam_seed import load_siam_seed_manifest
from src.quantum.siam_model import SIAMLayout, project_to_sector
from src.quantum.siam_observables import (
    SIAM_OBSERVABLES_V2,
    screening_observable_operators,
)
from src.quantum.siam_reference import (
    SIAM_REFERENCE_V2,
    propagate_sector,
    sector_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


SIAM_EXACT_REFERENCE_STORE_V1 = "siam_exact_reference_store_v1"
SIAM_EXACT_REFERENCE_STORE_V2 = "siam_exact_reference_store_v2"
SIAM_EXACT_PROPAGATOR_V1 = "siam_exact_sector_eigh_v1"
SIAM_EXACT_PREFIX_VIEW_V1 = "siam_exact_reference_prefix_view_v1"
SIAM_EXACT_PREFIX_VIEW_V2 = "siam_exact_reference_prefix_view_v2"
SIAM_EXACT_SLICE_VIEW_V1 = "siam_exact_reference_slice_view_v1"
SIAM_EXACT_SLICE_VIEW_V2 = "siam_exact_reference_slice_view_v2"

_REFERENCE_IDENTITY_FIELDS_V1 = (
    "schema",
    "propagator_schema",
    "reporting_only",
    "seed_sha256",
    "physical",
    "times",
)
_REFERENCE_IDENTITY_FIELDS_V2 = (
    "schema",
    "reference_schema",
    "observable_schema",
    "propagator_schema",
    "reporting_only",
    "seed_sha256",
    "physical",
    "times",
)


def _reference_identity_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the exact key identity without reinterpreting its schema."""

    schema = manifest.get("schema")
    if schema == SIAM_EXACT_REFERENCE_STORE_V1:
        fields = _REFERENCE_IDENTITY_FIELDS_V1
    elif schema == SIAM_EXACT_REFERENCE_STORE_V2:
        fields = _REFERENCE_IDENTITY_FIELDS_V2
    else:
        raise ValueError("SIAM full-reference manifest has an unsupported schema.")
    return {field: manifest.get(field) for field in fields}


def _view_schema(*, reference_schema: Any, kind: str) -> str:
    if reference_schema == SIAM_EXACT_REFERENCE_STORE_V1:
        return (
            SIAM_EXACT_PREFIX_VIEW_V1
            if kind == "prefix"
            else SIAM_EXACT_SLICE_VIEW_V1
        )
    if reference_schema == SIAM_EXACT_REFERENCE_STORE_V2:
        return (
            SIAM_EXACT_PREFIX_VIEW_V2
            if kind == "prefix"
            else SIAM_EXACT_SLICE_VIEW_V2
        )
    raise ValueError("SIAM exact-reference view source has an unsupported schema.")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(
        _canonical_bytes(
            {
                "dtype": value.dtype.str,
                "shape": [int(size) for size in value.shape],
            }
        )
    )
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _time_grid_sha256(times: Sequence[float]) -> str:
    return hashlib.sha256(
        _canonical_bytes([float(value) for value in times])
    ).hexdigest()


def _validated_reference_key(value: Any) -> str:
    key = str(value or "")
    if len(key) != 64 or any(
        character not in "0123456789abcdef" for character in key
    ):
        raise ValueError("SIAM exact-reference key must be a lowercase SHA-256 digest.")
    return key


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(
            f"Refusing to replace immutable SIAM exact-reference receipt: {path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(
                    "Refusing to replace immutable SIAM exact-reference receipt: "
                    f"{path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _validated_times(times: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in times)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("SIAM exact-reference times must be nonempty and finite.")
    if float(values[0]) != 0.0:
        raise ValueError("SIAM exact references must start at t=0.")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("SIAM exact-reference times must be strictly increasing.")
    return values


def _validated_slice_times(times: Sequence[float]) -> tuple[float, ...]:
    """Validate an ordered view grid without requiring it to start at zero."""

    values = tuple(float(value) for value in times)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("SIAM exact-reference slice times must be nonempty and finite.")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(
            "SIAM exact-reference slice times must be strictly increasing."
        )
    return values


def _observable_array_name(name: str) -> str:
    return f"observable__{name}"


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an NPZ with fixed member ordering and timestamps.

    ``numpy.savez_compressed`` embeds the current ZIP timestamp, so identical
    arrays can produce different artifact bytes.  A content-addressed store
    needs byte-stable data as well as stable array digests.
    """

    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, array in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asarray(array),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


@dataclass(frozen=True)
class SIAMExactReferenceArtifact:
    """One fully verified exact-reference artifact."""

    key: str
    manifest_path: Path
    data_path: Path
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]

    @property
    def manifest_sha256(self) -> str:
        return _file_sha256(self.manifest_path)

    @property
    def data_sha256(self) -> str:
        return _file_sha256(self.data_path)

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self.arrays["time"], dtype=float)

    @property
    def sector_states(self) -> np.ndarray:
        return np.asarray(self.arrays["state_real"], dtype=float) + 1.0j * np.asarray(
            self.arrays["state_imag"], dtype=float
        )

    def observable(self, name: str) -> np.ndarray:
        return np.asarray(self.arrays[_observable_array_name(name)], dtype=float)


@dataclass(frozen=True)
class SIAMExactReferencePrefixView:
    """Verified in-memory prefix of one immutable full-horizon reference.

    A prefix view intentionally has no data path of its own.  Its arrays are
    slices of the re-verified full artifact and its receipt binds those slices
    back to the full content key and both full-artifact file digests.
    """

    source: SIAMExactReferenceArtifact
    arrays: Mapping[str, np.ndarray]
    receipt: Mapping[str, Any]
    receipt_path: Path | None = None

    @property
    def key(self) -> str:
        return str(self.source.key)

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self.source.manifest

    @property
    def manifest_path(self) -> Path:
        return self.source.manifest_path

    @property
    def data_path(self) -> Path:
        return self.source.data_path

    @property
    def manifest_sha256(self) -> str:
        return self.source.manifest_sha256

    @property
    def data_sha256(self) -> str:
        return self.source.data_sha256

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self.arrays["time"], dtype=float)

    @property
    def sector_states(self) -> np.ndarray:
        return np.asarray(self.arrays["state_real"], dtype=float) + 1.0j * np.asarray(
            self.arrays["state_imag"], dtype=float
        )

    def observable(self, name: str) -> np.ndarray:
        return np.asarray(self.arrays[_observable_array_name(name)], dtype=float)


@dataclass(frozen=True)
class SIAMExactReferenceSliceView:
    """Verified contiguous slice of one immutable full-horizon reference.

    Unlike :class:`SIAMExactReferencePrefixView`, a slice may start after
    ``t=0``.  It still owns no data file: every array is an in-memory view of a
    freshly re-verified full artifact, and its digest-bound receipt records the
    exact source indices and ordered grid.  This is the canonical route for a
    resumed segment such as the Nb4 ``t=1..5`` continuation.
    """

    source: SIAMExactReferenceArtifact
    arrays: Mapping[str, np.ndarray]
    receipt: Mapping[str, Any]
    receipt_path: Path | None = None

    @property
    def key(self) -> str:
        return str(self.source.key)

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self.source.manifest

    @property
    def manifest_path(self) -> Path:
        return self.source.manifest_path

    @property
    def data_path(self) -> Path:
        return self.source.data_path

    @property
    def manifest_sha256(self) -> str:
        return self.source.manifest_sha256

    @property
    def data_sha256(self) -> str:
        return self.source.data_sha256

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self.arrays["time"], dtype=float)

    @property
    def sector_states(self) -> np.ndarray:
        return np.asarray(self.arrays["state_real"], dtype=float) + 1.0j * np.asarray(
            self.arrays["state_imag"], dtype=float
        )

    def observable(self, name: str) -> np.ndarray:
        return np.asarray(self.arrays[_observable_array_name(name)], dtype=float)


class SIAMExactReferenceStore:
    """Build once, then verify identity, arrays, and digests on every read."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def get_or_build(
        self,
        *,
        seed_path: str | Path,
        times: Sequence[float],
    ) -> SIAMExactReferenceArtifact:
        seed = Path(seed_path)
        seed_sha256_before = _file_sha256(seed)
        manifest = load_siam_seed_manifest(seed)
        seed_sha256 = _file_sha256(seed)
        if seed_sha256 != seed_sha256_before:
            raise ValueError("SIAM seed changed while the exact reference was loading.")
        reporting_times = _validated_times(times)
        identity = {
            "schema": SIAM_EXACT_REFERENCE_STORE_V2,
            "reference_schema": SIAM_REFERENCE_V2,
            "observable_schema": SIAM_OBSERVABLES_V2,
            "propagator_schema": SIAM_EXACT_PROPAGATOR_V1,
            "reporting_only": True,
            "seed_sha256": seed_sha256,
            "physical": dict(manifest.physical),
            "times": [float(value) for value in reporting_times],
        }
        key = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        artifact_root = self.root / key
        manifest_path = artifact_root / "manifest.json"
        data_path = artifact_root / "reference.npz"
        if artifact_root.exists():
            return self._load_verified(
                key=key,
                identity=identity,
                manifest_path=manifest_path,
                data_path=data_path,
            )

        arrays = self._build_arrays(
            physical=manifest.physical,
            times=reporting_times,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".{key}.", dir=str(self.root))
        )
        try:
            temporary_data = temporary_root / "reference.npz"
            _write_deterministic_npz(temporary_data, arrays)
            with temporary_data.open("rb") as handle:
                os.fsync(handle.fileno())
            array_contract = {
                name: {
                    "dtype": np.asarray(value).dtype.str,
                    "shape": [int(size) for size in np.asarray(value).shape],
                    "sha256": _array_sha256(np.asarray(value)),
                }
                for name, value in sorted(arrays.items())
            }
            artifact_manifest = {
                **identity,
                "key": key,
                "data_file": temporary_data.name,
                "data_sha256": _file_sha256(temporary_data),
                "arrays": array_contract,
            }
            temporary_manifest = temporary_root / "manifest.json"
            with temporary_manifest.open("wb") as handle:
                handle.write(
                    json.dumps(
                        artifact_manifest,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.rename(temporary_root, artifact_root)
            except OSError:
                # A concurrent builder won.  Its immutable bytes are verified
                # below against the same content identity.
                if not artifact_root.exists():
                    raise
        finally:
            if temporary_root.exists():
                shutil.rmtree(temporary_root)
        return self._load_verified(
            key=key,
            identity=identity,
            manifest_path=manifest_path,
            data_path=data_path,
        )

    def prefix_view(
        self,
        artifact: SIAMExactReferenceArtifact,
        *,
        times: Sequence[float],
        receipt_path: str | Path | None = None,
    ) -> SIAMExactReferencePrefixView:
        """Return a verified prefix without building another exact artifact.

        ``artifact`` must be the full-horizon object returned by this store.
        The requested ordered grid must equal its leading grid exactly.  The
        method re-verifies the full NPZ and manifest before slicing, and an
        optional immutable receipt records the complete lineage.  In
        particular this is the route for the canonical Nb4 ``t<=1`` view of
        the single ``t=0..5`` reference; callers must not invoke
        :meth:`get_or_build` on the shorter grid.
        """

        expected_root = self.root / str(artifact.key)
        expected_manifest_path = expected_root / "manifest.json"
        expected_data_path = expected_root / "reference.npz"
        if artifact.manifest_path != expected_manifest_path or (
            artifact.data_path != expected_data_path
        ):
            raise ValueError(
                "SIAM prefix view must originate from this exact-reference store."
            )
        identity = _reference_identity_from_manifest(artifact.manifest)
        expected_key = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        if expected_key != artifact.key:
            raise ValueError("SIAM full-reference identity/key mismatch.")
        verified = self._load_verified(
            key=str(artifact.key),
            identity=identity,
            manifest_path=expected_manifest_path,
            data_path=expected_data_path,
        )
        prefix_times = _validated_times(times)
        full_times = np.asarray(verified.times, dtype=float)
        count = len(prefix_times)
        if (
            count > int(full_times.size)
            or not np.array_equal(
                np.asarray(prefix_times, dtype=float), full_times[:count]
            )
            or _time_grid_sha256(prefix_times)
            != _time_grid_sha256(full_times[:count])
        ):
            raise ValueError(
                "SIAM exact-reference view grid must be an exact leading prefix "
                "of the verified full grid."
            )
        arrays = {
            name: np.asarray(value)[:count]
            for name, value in verified.arrays.items()
        }
        prefix_contract = {
            name: {
                "dtype": value.dtype.str,
                "shape": [int(size) for size in value.shape],
                "sha256": _array_sha256(value),
            }
            for name, value in sorted(arrays.items())
        }
        source_contract = {
            "reference_key": str(verified.key),
            "manifest_path": str(verified.manifest_path.resolve()),
            "manifest_sha256": verified.manifest_sha256,
            "data_path": str(verified.data_path.resolve()),
            "data_sha256": verified.data_sha256,
            "full_grid_sha256": _time_grid_sha256(full_times),
            "full_point_count": int(full_times.size),
        }
        if verified.manifest.get("schema") == SIAM_EXACT_REFERENCE_STORE_V2:
            source_contract.update(
                {
                    "reference_store_schema": SIAM_EXACT_REFERENCE_STORE_V2,
                    "reference_schema": str(verified.manifest["reference_schema"]),
                    "observable_schema": str(verified.manifest["observable_schema"]),
                }
            )
        receipt_core = {
            "schema": _view_schema(
                reference_schema=verified.manifest.get("schema"), kind="prefix"
            ),
            "reporting_only": True,
            "materialized_data_file": None,
            "source": source_contract,
            "prefix": {
                "times": [float(value) for value in prefix_times],
                "ordered_values_sha256": _time_grid_sha256(prefix_times),
                "point_count": int(count),
                "t_initial": float(prefix_times[0]),
                "t_final": float(prefix_times[-1]),
            },
            "arrays": prefix_contract,
        }
        receipt = {
            **receipt_core,
            "receipt_sha256": hashlib.sha256(
                _canonical_bytes(receipt_core)
            ).hexdigest(),
        }
        destination = None if receipt_path is None else Path(receipt_path)
        if destination is not None:
            _write_immutable_json(destination, receipt)
        return SIAMExactReferencePrefixView(
            source=verified,
            arrays=arrays,
            receipt=receipt,
            receipt_path=destination,
        )

    def slice_view(
        self,
        artifact: SIAMExactReferenceArtifact,
        *,
        times: Sequence[float],
        receipt_path: str | Path | None = None,
    ) -> SIAMExactReferenceSliceView:
        """Return a verified contiguous slice without creating another artifact.

        The requested ordered grid must occur exactly and contiguously in the
        source grid.  No approximate matching or interpolation is permitted.
        The full manifest, NPZ digest, and every array digest are re-verified
        before the slice is exposed.
        """

        expected_root = self.root / str(artifact.key)
        expected_manifest_path = expected_root / "manifest.json"
        expected_data_path = expected_root / "reference.npz"
        if artifact.manifest_path != expected_manifest_path or (
            artifact.data_path != expected_data_path
        ):
            raise ValueError(
                "SIAM slice view must originate from this exact-reference store."
            )
        identity = _reference_identity_from_manifest(artifact.manifest)
        expected_key = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        if expected_key != artifact.key:
            raise ValueError("SIAM full-reference identity/key mismatch.")
        verified = self._load_verified(
            key=str(artifact.key),
            identity=identity,
            manifest_path=expected_manifest_path,
            data_path=expected_data_path,
        )
        slice_times = _validated_slice_times(times)
        full_times = np.asarray(verified.times, dtype=float)
        requested = np.asarray(slice_times, dtype=float)
        matching_starts = np.flatnonzero(full_times == requested[0])
        if int(matching_starts.size) != 1:
            raise ValueError(
                "SIAM exact-reference slice grid does not have a unique exact "
                "start on the verified full grid."
            )
        start = int(matching_starts[0])
        stop = start + int(requested.size)
        if (
            stop > int(full_times.size)
            or not np.array_equal(
                requested,
                full_times[start:stop],
            )
            or _time_grid_sha256(slice_times)
            != _time_grid_sha256(full_times[start:stop])
        ):
            raise ValueError(
                "SIAM exact-reference slice grid must be an exact contiguous "
                "slice of the verified full grid."
            )
        arrays = {
            name: np.asarray(value)[start:stop]
            for name, value in verified.arrays.items()
        }
        array_contract = {
            name: {
                "dtype": value.dtype.str,
                "shape": [int(size) for size in value.shape],
                "sha256": _array_sha256(value),
            }
            for name, value in sorted(arrays.items())
        }
        source_contract = {
            "reference_key": str(verified.key),
            "manifest_path": str(verified.manifest_path.resolve()),
            "manifest_sha256": verified.manifest_sha256,
            "data_path": str(verified.data_path.resolve()),
            "data_sha256": verified.data_sha256,
            "full_grid_sha256": _time_grid_sha256(full_times),
            "full_point_count": int(full_times.size),
        }
        if verified.manifest.get("schema") == SIAM_EXACT_REFERENCE_STORE_V2:
            source_contract.update(
                {
                    "reference_store_schema": SIAM_EXACT_REFERENCE_STORE_V2,
                    "reference_schema": str(verified.manifest["reference_schema"]),
                    "observable_schema": str(verified.manifest["observable_schema"]),
                }
            )
        receipt_core = {
            "schema": _view_schema(
                reference_schema=verified.manifest.get("schema"), kind="slice"
            ),
            "reporting_only": True,
            "materialized_data_file": None,
            "source": source_contract,
            "slice": {
                "times": [float(value) for value in slice_times],
                "ordered_values_sha256": _time_grid_sha256(slice_times),
                "point_count": int(requested.size),
                "start_index": start,
                "stop_index_exclusive": stop,
                "t_initial": float(slice_times[0]),
                "t_final": float(slice_times[-1]),
            },
            "arrays": array_contract,
        }
        receipt = {
            **receipt_core,
            "receipt_sha256": hashlib.sha256(
                _canonical_bytes(receipt_core)
            ).hexdigest(),
        }
        destination = None if receipt_path is None else Path(receipt_path)
        if destination is not None:
            _write_immutable_json(destination, receipt)
        return SIAMExactReferenceSliceView(
            source=verified,
            arrays=arrays,
            receipt=receipt,
            receipt_path=destination,
        )

    def load_prefix_view(
        self,
        receipt_path: str | Path,
    ) -> SIAMExactReferencePrefixView:
        """Re-verify a stored prefix receipt and its one full source artifact."""

        source_path = Path(receipt_path)
        try:
            receipt = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("SIAM exact-prefix receipt is invalid JSON.") from exc
        if not isinstance(receipt, dict) or receipt.get("schema") not in {
            SIAM_EXACT_PREFIX_VIEW_V1,
            SIAM_EXACT_PREFIX_VIEW_V2,
        }:
            raise ValueError("SIAM exact-prefix receipt has the wrong schema.")
        receipt_core = dict(receipt)
        digest = receipt_core.pop("receipt_sha256", None)
        if digest != hashlib.sha256(_canonical_bytes(receipt_core)).hexdigest():
            raise ValueError("SIAM exact-prefix receipt digest mismatch.")
        source = receipt.get("source")
        prefix = receipt.get("prefix")
        if not isinstance(source, Mapping) or not isinstance(prefix, Mapping):
            raise ValueError("SIAM exact-prefix receipt is incomplete.")
        key = _validated_reference_key(source.get("reference_key"))
        artifact_root = self.root / key
        manifest_path = artifact_root / "manifest.json"
        data_path = artifact_root / "reference.npz"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("SIAM full-reference manifest is unavailable.") from exc
        identity = _reference_identity_from_manifest(manifest)
        if hashlib.sha256(_canonical_bytes(identity)).hexdigest() != key:
            raise ValueError("SIAM full-reference identity/key mismatch.")
        full = self._load_verified(
            key=key,
            identity=identity,
            manifest_path=manifest_path,
            data_path=data_path,
        )
        regenerated = self.prefix_view(full, times=prefix.get("times") or ())
        if dict(regenerated.receipt) != receipt:
            raise ValueError("SIAM exact-prefix receipt/source mismatch.")
        return SIAMExactReferencePrefixView(
            source=regenerated.source,
            arrays=regenerated.arrays,
            receipt=receipt,
            receipt_path=source_path,
        )

    def load_slice_view(
        self,
        receipt_path: str | Path,
    ) -> SIAMExactReferenceSliceView:
        """Re-verify a stored slice receipt and its one full source artifact."""

        source_path = Path(receipt_path)
        try:
            receipt = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("SIAM exact-slice receipt is invalid JSON.") from exc
        if not isinstance(receipt, dict) or receipt.get("schema") not in {
            SIAM_EXACT_SLICE_VIEW_V1,
            SIAM_EXACT_SLICE_VIEW_V2,
        }:
            raise ValueError("SIAM exact-slice receipt has the wrong schema.")
        receipt_core = dict(receipt)
        digest = receipt_core.pop("receipt_sha256", None)
        if digest != hashlib.sha256(_canonical_bytes(receipt_core)).hexdigest():
            raise ValueError("SIAM exact-slice receipt digest mismatch.")
        source = receipt.get("source")
        slice_contract = receipt.get("slice")
        if not isinstance(source, Mapping) or not isinstance(
            slice_contract, Mapping
        ):
            raise ValueError("SIAM exact-slice receipt is incomplete.")
        key = _validated_reference_key(source.get("reference_key"))
        artifact_root = self.root / key
        manifest_path = artifact_root / "manifest.json"
        data_path = artifact_root / "reference.npz"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("SIAM full-reference manifest is unavailable.") from exc
        identity = _reference_identity_from_manifest(manifest)
        if hashlib.sha256(_canonical_bytes(identity)).hexdigest() != key:
            raise ValueError("SIAM full-reference identity/key mismatch.")
        full = self._load_verified(
            key=key,
            identity=identity,
            manifest_path=manifest_path,
            data_path=data_path,
        )
        regenerated = self.slice_view(
            full,
            times=slice_contract.get("times") or (),
        )
        if dict(regenerated.receipt) != receipt:
            raise ValueError("SIAM exact-slice receipt/source mismatch.")
        return SIAMExactReferenceSliceView(
            source=regenerated.source,
            arrays=regenerated.arrays,
            receipt=receipt,
            receipt_path=source_path,
        )

    @staticmethod
    def _build_arrays(
        *,
        physical: Mapping[str, Any],
        times: Sequence[float],
    ) -> dict[str, np.ndarray]:
        n_bath = int(physical["n_bath"])
        u = float(physical["u"])
        v = float(physical["v_post_quench"])
        hopping = float(physical["hopping"])
        states = np.asarray(
            propagate_sector(
                n_bath,
                u=u,
                v=v,
                hopping=hopping,
                times=times,
                method="eigh",
            ),
            dtype=np.dtype("<c16"),
        )
        h_sector = sector_hamiltonian(
            n_bath,
            u=u,
            v=v,
            hopping=hopping,
        )
        operators = {
            name: project_to_sector(
                np.asarray(hamiltonian_matrix(operator), dtype=complex),
                n_bath,
            )
            for name, operator in screening_observable_operators(
                SIAMLayout(n_bath), v=v
            ).items()
        }
        arrays: dict[str, np.ndarray] = {
            "time": np.asarray(times, dtype=np.dtype("<f8")),
            "state_real": np.asarray(states.real, dtype=np.dtype("<f8")),
            "state_imag": np.asarray(states.imag, dtype=np.dtype("<f8")),
            "energy": np.asarray(
                [float(np.real(np.vdot(state, h_sector @ state))) for state in states],
                dtype=np.dtype("<f8"),
            ),
        }
        for name, operator in sorted(operators.items()):
            arrays[_observable_array_name(name)] = np.asarray(
                [float(np.real(np.vdot(state, operator @ state))) for state in states],
                dtype=np.dtype("<f8"),
            )
        if not all(np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("SIAM exact reference produced nonfinite arrays.")
        return arrays

    @staticmethod
    def _load_verified(
        *,
        key: str,
        identity: Mapping[str, Any],
        manifest_path: Path,
        data_path: Path,
    ) -> SIAMExactReferenceArtifact:
        if not manifest_path.is_file() or not data_path.is_file():
            raise ValueError(f"Incomplete SIAM exact-reference artifact: {key}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("SIAM exact-reference manifest is invalid JSON.") from exc
        for field, expected in identity.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"SIAM exact-reference manifest mismatch for {field}."
                )
        if manifest.get("key") != key:
            raise ValueError("SIAM exact-reference content key mismatch.")
        if manifest.get("data_file") != data_path.name:
            raise ValueError("SIAM exact-reference data filename mismatch.")
        if manifest.get("data_sha256") != _file_sha256(data_path):
            raise ValueError("SIAM exact-reference data digest mismatch.")
        array_contract = manifest.get("arrays")
        if not isinstance(array_contract, dict) or not array_contract:
            raise ValueError("SIAM exact-reference array contract is missing.")
        with np.load(data_path, allow_pickle=False) as archive:
            if set(archive.files) != set(array_contract):
                raise ValueError("SIAM exact-reference array schema mismatch.")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        for name, contract in array_contract.items():
            array = np.asarray(arrays[name])
            if array.dtype.str != contract.get("dtype"):
                raise ValueError(f"SIAM exact-reference dtype mismatch for {name}.")
            if list(array.shape) != contract.get("shape"):
                raise ValueError(f"SIAM exact-reference shape mismatch for {name}.")
            if _array_sha256(array) != contract.get("sha256"):
                raise ValueError(f"SIAM exact-reference array digest mismatch for {name}.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"SIAM exact-reference array {name} is nonfinite.")
        required = {"time", "state_real", "state_imag", "energy"}
        n_bath = int(identity["physical"]["n_bath"])
        required.update(_observable_array_name(name) for name in ("m_d", "D_d", "I_d"))
        if identity.get("schema") == SIAM_EXACT_REFERENCE_STORE_V2:
            if identity.get("observable_schema") != SIAM_OBSERVABLES_V2 or (
                identity.get("reference_schema") != SIAM_REFERENCE_V2
            ):
                raise ValueError("SIAM exact-reference reporting schema mismatch.")
            required.update(
                _observable_array_name(name) for name in ("N_d", "S_d2")
            )
        required.update(
            _observable_array_name(f"{prefix}_{site}")
            for prefix in ("C", "Sigma")
            for site in range(1, n_bath + 1)
        )
        if set(arrays) != required:
            raise ValueError("SIAM exact-reference observable schema mismatch.")
        if identity.get("schema") == SIAM_EXACT_REFERENCE_STORE_V2:
            n_d = np.asarray(arrays[_observable_array_name("N_d")], dtype=float)
            d_d = np.asarray(arrays[_observable_array_name("D_d")], dtype=float)
            s_d2 = np.asarray(arrays[_observable_array_name("S_d2")], dtype=float)
            if not np.allclose(
                s_d2, 0.75 * (n_d - 2.0 * d_d), rtol=0.0, atol=1.0e-12
            ):
                raise ValueError("SIAM exact-reference S_d2 identity mismatch.")
            cumulative = s_d2.copy()
            for site in range(1, n_bath + 1):
                cumulative = cumulative + np.asarray(
                    arrays[_observable_array_name(f"C_{site}")], dtype=float
                )
                sigma = np.asarray(
                    arrays[_observable_array_name(f"Sigma_{site}")], dtype=float
                )
                if not np.allclose(sigma, cumulative, rtol=0.0, atol=1.0e-12):
                    raise ValueError(
                        f"SIAM exact-reference Sigma_{site} identity mismatch."
                    )
        times = np.asarray(identity["times"], dtype=float)
        if not np.array_equal(np.asarray(arrays["time"], dtype=float), times):
            raise ValueError("SIAM exact-reference reporting grid mismatch.")
        count = int(times.size)
        sector_dimension = int(identity["physical"]["sector_dimension"])
        if arrays["state_real"].shape != (count, sector_dimension):
            raise ValueError("SIAM exact-reference state shape mismatch.")
        for name, array in arrays.items():
            if name not in {"state_real", "state_imag"} and array.shape != (count,):
                raise ValueError(f"SIAM exact-reference series shape mismatch for {name}.")
        return SIAMExactReferenceArtifact(
            key=key,
            manifest_path=manifest_path,
            data_path=data_path,
            manifest=manifest,
            arrays=arrays,
        )


__all__ = [
    "SIAM_EXACT_PROPAGATOR_V1",
    "SIAM_EXACT_PREFIX_VIEW_V1",
    "SIAM_EXACT_PREFIX_VIEW_V2",
    "SIAM_EXACT_REFERENCE_STORE_V1",
    "SIAM_EXACT_REFERENCE_STORE_V2",
    "SIAM_EXACT_SLICE_VIEW_V1",
    "SIAM_EXACT_SLICE_VIEW_V2",
    "SIAMExactReferenceArtifact",
    "SIAMExactReferencePrefixView",
    "SIAMExactReferenceSliceView",
    "SIAMExactReferenceStore",
]
