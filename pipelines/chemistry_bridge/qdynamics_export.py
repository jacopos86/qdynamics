"""Adapter from Doron Wu's qdynamics ``<prefix>_ele.h5`` export to the strict intake.

Pinned producer (primary source, read read-only over the GitHub API; nothing was
cloned, installed, imported or executed):

    repository  https://github.com/jacopos86/qdynamics
    branch      doron-chemistry
    commit      e5ec5bc45d02169e77ce93b05126c9804a03cdb6
                (author doronwu <dwu269@wisc.edu>, 2026-08-26T21:02:57Z, "tested-eph")
    license     GPL-3.0 (repository metadata)
    writer      src/chemistry/pyscf/read_write_pyscf_results.py :: write_matrix_elements_h5
                (git blob 3bf71a4d0d2c)
    driver      src/chemistry/pyscf/main_pyscf.py (git blob 594293ae21d9)
    integrals   src/chemistry/pyscf/electron_solver.py :: PySCFDriver.get_matrix_elements
                (git blob 8400fc64c5ac)

What that driver writes (``<prefix>_ele.h5``):

    metadata (group attrs)  mol_str, basis, method, charge, spin, unit, xc, e_scf,
                            converged, nelec, nmo, ms2, e_converg, d_converg, max_iter,
                            isotope_masses, h1_convention='spin_orbital_interleaved',
                            h2_convention='chemist_spin_orbital_coulomb',
                            h2_antisymmetrized=False
    h1  (2n, 2n)            interleaved spin orbitals (a0, b0, a1, b1, ...):
                            h1[0::2, 0::2] = h1[1::2, 1::2] = h_mo, other blocks zero
    h2  (2n, 2n, 2n, 2n)    chemists' (ij|kl) Coulomb integrals, NOT antisymmetrized:
                            h2[s::2, s::2, s'::2, s'::2] = eri_mo for s, s' in {0, 1},
                            every other entry zero
    nuclear_repulsion       scalar ``mol.energy_nuc()`` (nuclear only, no frozen core)
    mo_coeff, mo_energy, mo_occ   optional datasets

At that commit the PySCF job writes no FCIDUMP (``write_fcidump`` is parsed by
``input_parser.py`` but never consumed by ``main_pyscf.py``).  The FCIDUMP
reader below is therefore a secondary input path for a future export, not the
producer's current output.

The repository's consumer contract is ``RestrictedClosedShellMolecularProblem``:
real spatial-MO ``h_pq`` and chemists' ``(pq|rs)`` over orthonormal spatial
orbitals, blocked alpha-then-beta Jordan-Wigner order, one explicit scalar.
This module converts, checks every convention it can check numerically, and
binds the conventions it cannot check (notation, units, orbital frame) by
declaration and provenance.  A numerical symmetry check is necessary, not
sufficient: a physicists'-ordered tensor written with chemists' labels passes
the same symmetry tests, so notation is bound by the declared producer
convention, never inferred from symmetry alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.chemistry_bridge import BRIDGE_SCHEMA_VERSION
from pipelines.static_adapt.builders.molecular_diagnostic_intake import (
    MolecularDiagnosticDeclaration,
)
from src.quantum.chemistry.psi4_adapter import RestrictedClosedShellMolecularProblem


PINNED_PRODUCER: dict[str, Any] = {
    "repository": "https://github.com/jacopos86/qdynamics",
    "branch": "doron-chemistry",
    "commit": "e5ec5bc45d02169e77ce93b05126c9804a03cdb6",
    "author": "doronwu <dwu269@wisc.edu>",
    "commit_date_utc": "2026-08-26T21:02:57Z",
    "license_spdx": "GPL-3.0",
    "writer": "src/chemistry/pyscf/read_write_pyscf_results.py::write_matrix_elements_h5",
    "writer_blob": "3bf71a4d0d2c",
    "driver": "src/chemistry/pyscf/main_pyscf.py",
    "driver_blob": "594293ae21d9",
    "integrals": "src/chemistry/pyscf/electron_solver.py::PySCFDriver.get_matrix_elements",
    "integrals_blob": "8400fc64c5ac",
    "fcidump_written_by_driver": False,
}

QDYNAMICS_ELE_H5_SCHEMA = "qdynamics_matrix_elements_ele_h5"
CONTAINER_IN_MEMORY = "in_memory_mapping"
ORIGIN_SYNTHETIC = "synthetic"
ORIGIN_SIMULATED = "simulated"
ORIGIN_UNKNOWN = "unknown"
ORIGIN_ATTESTED = "attested"
ORIGIN_CHOICES = (ORIGIN_SYNTHETIC, ORIGIN_SIMULATED, ORIGIN_UNKNOWN, ORIGIN_ATTESTED)
BRIDGE_PROBLEM_SCHEMA = "doron_qdynamics_bridge_problem_v1"
BRIDGE_MANIFEST_SCHEMA = "doron_qdynamics_bridge_manifest_v1"

EXPECTED_H1_CONVENTION = "spin_orbital_interleaved"
EXPECTED_H2_CONVENTION = "chemist_spin_orbital_coulomb"

# Declaration values accepted by the strict intake (its only accepted set).
INTAKE_ENERGY_UNITS = "hartree"
INTAKE_ERI_CONVENTION = "real_spatial_chemists"
INTAKE_ORBITAL_BASIS = "orthonormal_spatial_mo"
INTAKE_OCCUPIED_ORDER = "occupied_prefix"


class BridgeDependencyUnavailable(RuntimeError):
    """An optional dependency needed by one bridge path is not importable."""

    def __init__(self, dependency: str, message: str) -> None:
        super().__init__(message)
        self.dependency = str(dependency)


class QdynamicsConventionError(ValueError):
    """A producer convention or numerical witness failed, with a named code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = str(code)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return _sha256_bytes(json.dumps(list(canonical.shape)).encode() + canonical.tobytes(order="C"))


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _as_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"true", "1", "yes"}:
            return True
        if key in {"false", "0", "no"}:
            return False
    if isinstance(value, Integral):
        return bool(int(value))
    raise QdynamicsConventionError("metadata_type", f"{label} must be a boolean; got {value!r}.")


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise QdynamicsConventionError("metadata_type", f"{label} must be an integer, not a boolean.")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise QdynamicsConventionError("metadata_type", f"{label} must be an integer; got {value!r}.")


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise QdynamicsConventionError("metadata_type", f"{label} must be a real number, not a boolean.")
    if isinstance(value, Real):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError as exc:
            raise QdynamicsConventionError("metadata_type", f"{label} must be a real number; got {value!r}.") from exc
    else:
        raise QdynamicsConventionError("metadata_type", f"{label} must be a real number; got {value!r}.")
    if not math.isfinite(result):
        raise QdynamicsConventionError("metadata_nonfinite", f"{label} must be finite.")
    return result


def _scalar_dataset(value: Any, label: str) -> float:
    """Strict scalar dataset: a real, finite, zero-dimensional number.

    The pinned producer writes ``nuclear_repulsion`` as a scalar dataset
    (``create_dataset(..., data=float(...))``).  Anything with a shape, a
    string, a boolean or a nonfinite value is rejected before a value is
    extracted; nothing is truncated or coerced.
    """

    if isinstance(value, (bool, np.bool_)):
        raise QdynamicsConventionError("nuclear_repulsion_type", f"{label} must be a real number, not a boolean.")
    if isinstance(value, (str, bytes)):
        raise QdynamicsConventionError("nuclear_repulsion_type", f"{label} must be a numeric scalar dataset, not text.")
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise QdynamicsConventionError("nuclear_repulsion_type", f"{label} is not numeric: {exc}") from exc
    if array.ndim != 0:
        raise QdynamicsConventionError("nuclear_repulsion_shape", f"{label} must be a scalar dataset; got shape {array.shape}.")
    if array.dtype.kind == "c":
        if float(np.abs(array.imag)) != 0.0:
            raise QdynamicsConventionError("complex_tensor", f"{label} has a nonzero imaginary part.")
        array = array.real
    if array.dtype.kind not in {"f", "i", "u"}:
        raise QdynamicsConventionError("nuclear_repulsion_type", f"{label} must be a real numeric scalar; got dtype {array.dtype}.")
    result = float(array)
    if not math.isfinite(result):
        raise QdynamicsConventionError("nuclear_repulsion_nonfinite", f"{label} must be finite; got {result!r}.")
    return result


def _real_array(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind == "c":
        if np.max(np.abs(array.imag)) != 0.0:
            raise QdynamicsConventionError("complex_tensor", f"{label} has nonzero imaginary entries; real orbitals are required.")
        array = array.real
    if array.dtype.kind not in {"f", "i", "u"}:
        raise QdynamicsConventionError("tensor_dtype", f"{label} must be a real numeric array; got dtype {array.dtype}.")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise QdynamicsConventionError("tensor_nonfinite", f"{label} contains nonfinite values.")
    return array


@dataclass(frozen=True)
class QdynamicsMatrixElements:
    """In-memory mirror of ``read_write_pyscf_results.load_h5`` for ``<prefix>_ele.h5``."""

    metadata: dict[str, Any]
    h1: np.ndarray
    h2: np.ndarray
    nuclear_repulsion: float
    mo_occ: np.ndarray | None = None
    mo_energy: np.ndarray | None = None
    source: dict[str, Any] = field(default_factory=dict)


def validate_attestation(
    attestation: Any,
    *,
    digest_key: str,
) -> dict[str, Any]:
    """Validate an attestation record's actual representation, without coercion.

    ``attested_by`` must be a ``str`` (not a bool or any other type) that is not
    blank; its value is preserved exactly.  ``digest_key`` must name a 64-hex
    SHA-256 string.  A valid record is still only a claim by the named attester;
    nothing here authenticates a producer.
    """

    if not isinstance(attestation, Mapping):
        raise QdynamicsConventionError("origin_attestation", f"attestation must be a mapping with 'attested_by' and '{digest_key}'; got {type(attestation).__name__}.")
    who = attestation.get("attested_by", None)
    if isinstance(who, bool) or not isinstance(who, str) or not who.strip():
        raise QdynamicsConventionError("origin_attestation", f"attestation.attested_by must be a nonblank string identity; got {who!r}.")
    digest = attestation.get(digest_key, None)
    if isinstance(digest, bool) or not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise QdynamicsConventionError("origin_attestation", f"attestation.{digest_key} must be a 64-character SHA-256 hex string; got {digest!r}.")
    return {str(k): v for k, v in attestation.items()}


def _origin_record(
    *,
    container_schema: str,
    declared_origin: str | None,
    declared_producer: str | None,
    attestation: Mapping[str, Any] | None,
    caller_source: Mapping[str, Any],
    file_sha256: str | None,
) -> dict[str, Any]:
    """Keep three things apart: observed container, claimed producer, established origin.

    * ``container_schema`` is what the bytes look like (set by the loader that
      produced them, never by the caller).
    * ``claimed_producer`` is a claim made by whoever supplied the data.
    * ``origin`` is ``synthetic`` / ``simulated`` / ``unknown`` / ``attested``.
      It defaults to ``unknown``; a hash or a recognized schema never promotes
      it.  ``attested`` requires an attestation record naming who attests and
      the file hash it covers, and is still a recorded claim, not proof.
    * Caller flags ``simulation=True`` or ``producer_generated=False`` force a
      non-producer origin even if a producer schema or name is present.
    """

    origin = ORIGIN_UNKNOWN if declared_origin in {None, ""} else str(declared_origin).strip().lower()
    if origin not in ORIGIN_CHOICES:
        raise QdynamicsConventionError("origin_declaration", f"declared_origin must be one of {ORIGIN_CHOICES}; got {declared_origin!r}.")
    simulation = caller_source.get("simulation")
    producer_generated = caller_source.get("producer_generated")
    forced_reason = None
    if simulation is True or (isinstance(simulation, str) and simulation.strip().lower() in {"true", "yes", "1"}):
        forced_reason = "caller_source.simulation=True"
        origin = ORIGIN_SIMULATED if origin in {ORIGIN_UNKNOWN, ORIGIN_ATTESTED, ORIGIN_SIMULATED} else origin
    if producer_generated is False or (isinstance(producer_generated, str) and producer_generated.strip().lower() in {"false", "no", "0"}):
        forced_reason = (forced_reason + "; " if forced_reason else "") + "caller_source.producer_generated=False"
        if origin == ORIGIN_ATTESTED:
            origin = ORIGIN_SIMULATED
        if origin == ORIGIN_UNKNOWN:
            origin = ORIGIN_SIMULATED
    attestation_record: dict[str, Any] | None = None
    if origin == ORIGIN_ATTESTED:
        if attestation is None:
            raise QdynamicsConventionError("origin_attestation", "origin='attested' requires attestation={'attested_by': <who>, 'file_sha256': <sha256>, ...}.")
        attestation_record = validate_attestation(attestation, digest_key="file_sha256")
        if file_sha256 is None:
            raise QdynamicsConventionError("origin_attestation", "an attested origin needs a file whose sha256 the attestation covers; in-memory mappings cannot be attested.")
        if str(attestation_record["file_sha256"]).lower() != str(file_sha256).lower():
            raise QdynamicsConventionError("origin_attestation", "attestation.file_sha256 does not match the loaded file.")
    elif attestation:
        raise QdynamicsConventionError("origin_attestation", f"attestation supplied but declared_origin is {origin!r}; declare origin='attested' explicitly.")
    return {
        "container_schema": str(container_schema),
        "schema_reference_producer": dict(PINNED_PRODUCER),
        "claimed_producer": None if declared_producer in {None, ""} else str(declared_producer),
        "origin": origin,
        "origin_attestation": attestation_record,
        "origin_forced_by": forced_reason,
        "producer_origin_established": False,
        "origin_note": "schema recognition and content hashes identify bytes, not who produced them",
    }


def qdynamics_matrix_elements_from_mapping(
    mapping: Mapping[str, Any],
    *,
    source: Mapping[str, Any] | None = None,
    declared_origin: str | None = None,
    declared_producer: str | None = None,
) -> QdynamicsMatrixElements:
    """Build the in-memory record from a ``load_h5``-shaped mapping.

    ``mapping`` carries ``metadata`` (attrs), ``h1``, ``h2``, ``nuclear_repulsion``
    and optionally ``mo_occ``/``mo_energy``/``mo_coeff``.  Extra keys are recorded
    in ``source['extra_datasets']`` and otherwise ignored.  ``source`` is the
    caller's description of where the mapping came from; it is recorded under
    ``caller_source`` and cannot set the container schema.  Origin is declared
    (``declared_origin``) and defaults to ``unknown``; an in-memory mapping can
    never be ``attested`` because there is no file for an attestation to cover,
    so this public entry point takes no attestation and no container argument.
    """

    return _matrix_elements_from_payload(
        mapping,
        caller_source=source,
        declared_origin=declared_origin,
        declared_producer=declared_producer,
        attestation=None,
        container_schema=CONTAINER_IN_MEMORY,
        file_sha256=None,
    )


def _matrix_elements_from_payload(
    mapping: Mapping[str, Any],
    *,
    caller_source: Mapping[str, Any] | None,
    declared_origin: str | None,
    declared_producer: str | None,
    attestation: Mapping[str, Any] | None,
    container_schema: str,
    file_sha256: str | None,
) -> QdynamicsMatrixElements:
    """Shared constructor.  Only ``load_qdynamics_matrix_elements_h5`` may name a
    file container, and it must pass the sha256 it computed over the file bytes."""

    if container_schema == QDYNAMICS_ELE_H5_SCHEMA and not (isinstance(file_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", file_sha256)):
        raise QdynamicsConventionError("container_claim", "a file container requires the loader's sha256 over the file bytes.")
    if container_schema != QDYNAMICS_ELE_H5_SCHEMA and file_sha256 is not None:
        raise QdynamicsConventionError("container_claim", "a file sha256 without a file container is not admissible.")

    if not isinstance(mapping, Mapping):
        raise QdynamicsConventionError("payload_type", "qdynamics payload must be a mapping.")
    missing = [key for key in ("metadata", "h1", "h2", "nuclear_repulsion") if key not in mapping]
    if missing:
        raise QdynamicsConventionError("missing_dataset", f"qdynamics payload is missing {missing}.")
    metadata_raw = mapping["metadata"]
    if not isinstance(metadata_raw, Mapping):
        raise QdynamicsConventionError("payload_type", "qdynamics metadata must be a mapping of attrs.")
    metadata = {str(key): _decode_attr(value) for key, value in metadata_raw.items()}
    h1 = _real_array(mapping["h1"], "h1")
    h2 = _real_array(mapping["h2"], "h2")
    nuclear = _scalar_dataset(mapping["nuclear_repulsion"], "nuclear_repulsion")
    mo_occ = None if mapping.get("mo_occ") is None else _real_array(mapping["mo_occ"], "mo_occ").reshape(-1)
    mo_energy = None if mapping.get("mo_energy") is None else _real_array(mapping["mo_energy"], "mo_energy").reshape(-1)
    extra = sorted(str(key) for key in mapping if key not in {"metadata", "h1", "h2", "nuclear_repulsion", "mo_occ", "mo_energy", "mo_coeff"})
    caller_record = {str(k): v for k, v in dict(caller_source or {}).items()}
    source_record: dict[str, Any] = {"caller_source": caller_record}
    source_record.update(_origin_record(
        container_schema=container_schema,
        declared_origin=declared_origin,
        declared_producer=declared_producer,
        attestation=attestation,
        caller_source=caller_record,
        file_sha256=file_sha256,
    ))
    source_record["datasets"] = {
        "h1": {"shape": list(h1.shape), "sha256": _array_sha256(h1)},
        "h2": {"shape": list(h2.shape), "sha256": _array_sha256(h2)},
        "nuclear_repulsion": {"value": float(nuclear)},
        "mo_occ": None if mo_occ is None else {"shape": list(mo_occ.shape), "sha256": _array_sha256(mo_occ)},
        "mo_energy": None if mo_energy is None else {"shape": list(mo_energy.shape), "sha256": _array_sha256(mo_energy)},
        "mo_coeff_present": mapping.get("mo_coeff") is not None,
    }
    source_record["extra_datasets"] = extra
    return QdynamicsMatrixElements(
        metadata=metadata,
        h1=h1,
        h2=h2,
        nuclear_repulsion=float(nuclear),
        mo_occ=mo_occ,
        mo_energy=mo_energy,
        source=source_record,
    )


def load_qdynamics_matrix_elements_h5(
    path: str | Path,
    *,
    declared_origin: str | None = None,
    declared_producer: str | None = None,
    attestation: Mapping[str, Any] | None = None,
) -> QdynamicsMatrixElements:
    """Read a ``<prefix>_ele.h5`` file in the pinned producer's schema.

    Requires ``h5py`` (not a repository dependency); raises
    ``BridgeDependencyUnavailable`` with the exact missing module otherwise.
    The whole file is hashed before any dataset is read.  Reading a file in
    this schema establishes the container only; its origin stays ``unknown``
    unless declared, and ``attested`` needs an attestation covering the file's
    sha256.
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"qdynamics export not found: {file_path}")
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise BridgeDependencyUnavailable(
            "h5py",
            "Reading Doron's HDF5 export requires h5py, which is not installed in this "
            "environment.  Install it in an isolated environment or convert the file "
            "with the producer's own h5 reader into a JSON mapping and use "
            "qdynamics_matrix_elements_from_mapping.",
        ) from exc
    file_sha256 = _sha256_file(file_path)
    payload: dict[str, Any] = {}
    dataset_dtypes: dict[str, str] = {}
    with h5py.File(file_path, "r") as handle:
        if "metadata" not in handle:
            raise QdynamicsConventionError("missing_metadata", "HDF5 file has no 'metadata' group.")
        payload["metadata"] = {str(key): _decode_attr(value) for key, value in handle["metadata"].attrs.items()}
        for key in handle.keys():
            if key == "metadata":
                continue
            dataset = handle[key]
            dataset_dtypes[str(key)] = str(dataset.dtype)
            payload[str(key)] = dataset[()]
    return _matrix_elements_from_payload(
        payload,
        caller_source={
            "path": str(file_path),
            "file_sha256": file_sha256,
            "file_size_bytes": int(file_path.stat().st_size),
            "dataset_dtypes": dataset_dtypes,
        },
        declared_origin=declared_origin,
        declared_producer=declared_producer,
        attestation=attestation,
        container_schema=QDYNAMICS_ELE_H5_SCHEMA,
        file_sha256=file_sha256,
    )


@dataclass(frozen=True)
class SpatialTensorExtraction:
    h_mo: np.ndarray
    eri_mo: np.ndarray
    checks: dict[str, float]


def spatial_tensors_from_interleaved(
    h1: np.ndarray,
    h2: np.ndarray,
    *,
    atol: float = 1.0e-12,
) -> SpatialTensorExtraction:
    """Extract spatial ``h_mo``/``eri_mo`` from the producer's interleaved spin tensors.

    Checks, each with a named failure code: shapes ``(2n, 2n)``/``(2n,)*4``;
    cross-spin ``h1`` blocks zero; alpha and beta ``h1`` blocks equal
    (restricted reference); the four spin-allowed ``h2`` placements equal to one
    spatial tensor; every spin-forbidden ``h2`` entry zero; ``h`` symmetric;
    the three generating real chemists' ERI symmetries.
    """

    tolerance = float(atol)
    h1 = _real_array(h1, "h1")
    h2 = _real_array(h2, "h2")
    if h1.ndim != 2 or h1.shape[0] != h1.shape[1] or h1.shape[0] % 2 != 0 or h1.shape[0] == 0:
        raise QdynamicsConventionError("h1_shape", f"h1 must be square with even dimension; got {h1.shape}.")
    nso = int(h1.shape[0])
    n = nso // 2
    if h2.shape != (nso, nso, nso, nso):
        raise QdynamicsConventionError("h2_shape", f"h2 must have shape {(nso,) * 4}; got {h2.shape}.")
    alpha = slice(0, nso, 2)
    beta = slice(1, nso, 2)
    h_alpha = h1[alpha, alpha]
    h_beta = h1[beta, beta]
    checks: dict[str, float] = {}
    checks["h1_cross_spin_max_abs"] = float(max(np.max(np.abs(h1[alpha, beta])), np.max(np.abs(h1[beta, alpha]))))
    checks["h1_alpha_beta_block_max_abs_diff"] = float(np.max(np.abs(h_alpha - h_beta)))
    eri = np.array(h2[alpha, alpha, alpha, alpha], dtype=np.float64, copy=True)
    placement_diff = 0.0
    for s_ij in (alpha, beta):
        for s_kl in (alpha, beta):
            placement_diff = max(placement_diff, float(np.max(np.abs(h2[s_ij, s_ij, s_kl, s_kl] - eri))))
    checks["h2_spin_block_placement_max_abs_diff"] = placement_diff
    spin = np.arange(nso) % 2
    allowed = (spin[:, None, None, None] == spin[None, :, None, None]) & (spin[None, None, :, None] == spin[None, None, None, :])
    forbidden_max = float(np.max(np.abs(h2[~allowed]))) if np.any(~allowed) else 0.0
    checks["h2_spin_forbidden_max_abs"] = forbidden_max
    checks["h_hermitian_max_abs"] = float(np.max(np.abs(h_alpha - h_alpha.T)))
    checks["eri_first_pair_max_abs"] = float(np.max(np.abs(eri - eri.transpose(1, 0, 2, 3))))
    checks["eri_second_pair_max_abs"] = float(np.max(np.abs(eri - eri.transpose(0, 1, 3, 2))))
    checks["eri_pair_exchange_max_abs"] = float(np.max(np.abs(eri - eri.transpose(2, 3, 0, 1))))
    failures = {
        "h1_cross_spin_max_abs": "h1_cross_spin_nonzero",
        "h1_alpha_beta_block_max_abs_diff": "h1_spin_blocks_differ",
        "h2_spin_block_placement_max_abs_diff": "h2_spin_block_placement",
        "h2_spin_forbidden_max_abs": "h2_spin_forbidden_nonzero",
        "h_hermitian_max_abs": "h_not_symmetric",
        "eri_first_pair_max_abs": "eri_symmetry",
        "eri_second_pair_max_abs": "eri_symmetry",
        "eri_pair_exchange_max_abs": "eri_symmetry",
    }
    for name, code in failures.items():
        if checks[name] > tolerance:
            raise QdynamicsConventionError(code, f"{name}={checks[name]:.3e} exceeds atol={tolerance:.1e}.")
    return SpatialTensorExtraction(h_mo=np.asarray(h_alpha, dtype=np.float64).copy(), eri_mo=eri, checks=checks)


def closed_shell_determinant_energy(scalar: float, h: np.ndarray, g: np.ndarray, n_occupied: int) -> float:
    """E_det = E_scalar + 2 sum_i h_ii + sum_ij (2 g_iijj - g_ijji) over the occupied prefix."""

    k = int(n_occupied)
    terms = [float(scalar)]
    terms.extend(2.0 * float(h[i, i]) for i in range(k))
    terms.extend(2.0 * float(g[i, i, j, j]) - float(g[i, j, j, i]) for i in range(k) for j in range(k))
    return math.fsum(terms)


@dataclass(frozen=True)
class FrozenCoreReduction:
    """Freeze the lowest ``n_frozen`` occupied spatial orbitals (occupied-prefix order)."""

    n_frozen: int

    def __post_init__(self) -> None:
        if int(self.n_frozen) < 0:
            raise ValueError("n_frozen must be non-negative.")


def frozen_core_contract(
    h: np.ndarray,
    g: np.ndarray,
    scalar: float,
    *,
    frozen: Sequence[int],
    active: Sequence[int],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Contract doubly occupied frozen orbitals into the scalar and one-body tensor.

    E_core = E_scalar + 2 sum_{c} h_cc + sum_{c,d} (2 (cc|dd) - (cd|dc))
    h'_pq  = h_pq + sum_c (2 (pq|cc) - (pc|cq))        for active p, q
    g'     = g restricted to the active orbitals.
    """

    h = _real_array(h, "h")
    g = _real_array(g, "g")
    frozen_idx = np.asarray([int(v) for v in frozen], dtype=int)
    active_idx = np.asarray([int(v) for v in active], dtype=int)
    if frozen_idx.size and active_idx.size and np.intersect1d(frozen_idx, active_idx).size:
        raise ValueError("frozen and active orbital sets overlap.")
    core_scalar = float(scalar)
    if frozen_idx.size:
        core_scalar += 2.0 * float(np.sum(h[frozen_idx, frozen_idx]))
        g_cc = g[np.ix_(frozen_idx, frozen_idx, frozen_idx, frozen_idx)]
        core_scalar += float(2.0 * np.einsum("ccdd->", g_cc) - np.einsum("cddc->", g_cc))
    h_active = np.array(h[np.ix_(active_idx, active_idx)], dtype=np.float64, copy=True)
    if frozen_idx.size and active_idx.size:
        g_pqcc = g[np.ix_(active_idx, active_idx, frozen_idx, frozen_idx)]
        g_pccq = g[np.ix_(active_idx, frozen_idx, frozen_idx, active_idx)]
        h_active += 2.0 * np.einsum("pqcc->pq", g_pqcc) - np.einsum("pccq->pq", g_pccq)
    g_active = np.array(g[np.ix_(active_idx, active_idx, active_idx, active_idx)], dtype=np.float64, copy=True)
    return core_scalar, h_active, g_active


@dataclass(frozen=True)
class BridgeProblem:
    """A converted problem plus the declaration and provenance the intake needs."""

    problem: RestrictedClosedShellMolecularProblem
    declaration: MolecularDiagnosticDeclaration
    provenance: dict[str, Any]
    checks: dict[str, Any]


def bridge_problem_from_qdynamics(
    elements: QdynamicsMatrixElements,
    *,
    frozen_core: FrozenCoreReduction | None = None,
    symmetry_atol: float = 1.0e-12,
    witness_atol: float = 1.0e-8,
) -> BridgeProblem:
    """Convert the producer's export into the repository's closed-shell problem.

    ``witness_atol`` bounds |E_det(tensors) - e_scf| here; the strict intake
    re-checks that witness at its own ``atol`` (1e-12 by default), so a
    producer whose SCF converged more loosely than that fails there, by design.
    """

    meta = dict(elements.metadata)
    checks: dict[str, Any] = {}
    h1_convention = str(meta.get("h1_convention", "")).strip()
    h2_convention = str(meta.get("h2_convention", "")).strip()
    if h1_convention != EXPECTED_H1_CONVENTION:
        raise QdynamicsConventionError("h1_convention", f"expected {EXPECTED_H1_CONVENTION!r}; got {h1_convention!r}.")
    if h2_convention != EXPECTED_H2_CONVENTION:
        raise QdynamicsConventionError("h2_convention", f"expected {EXPECTED_H2_CONVENTION!r}; got {h2_convention!r}.")
    if "h2_antisymmetrized" not in meta:
        raise QdynamicsConventionError("h2_antisymmetrized_missing", "metadata must declare h2_antisymmetrized.")
    if _as_bool(meta["h2_antisymmetrized"], "h2_antisymmetrized"):
        raise QdynamicsConventionError("h2_antisymmetrized", "antisymmetrized two-body tensors are not chemists' Coulomb integrals.")
    method = str(meta.get("method", "")).strip().upper()
    if method != "RHF":
        raise QdynamicsConventionError(
            "method",
            f"only method='RHF' is admissible (got {method!r}); a Kohn-Sham total energy is not a determinant-energy witness.",
        )
    spin = _as_int(meta.get("spin", 0), "spin")
    ms2 = _as_int(meta.get("ms2", spin), "ms2")
    if spin != 0 or ms2 != 0:
        raise QdynamicsConventionError("open_shell", f"closed-shell export required; spin={spin}, ms2={ms2}.")
    if "converged" not in meta:
        raise QdynamicsConventionError("converged_missing", "metadata must record SCF convergence.")
    if not _as_bool(meta["converged"], "converged"):
        raise QdynamicsConventionError("scf_not_converged", "producer recorded converged=False.")
    charge = _as_int(meta.get("charge", 0), "charge")
    nelec = _as_int(meta["nelec"], "nelec") if "nelec" in meta else None
    nmo_declared = _as_int(meta["nmo"], "nmo") if "nmo" in meta else None
    e_scf = _as_float(meta["e_scf"], "e_scf") if "e_scf" in meta else None
    if e_scf is None:
        raise QdynamicsConventionError("e_scf_missing", "metadata must carry e_scf for the determinant-energy witness.")
    if nelec is None:
        raise QdynamicsConventionError("nelec_missing", "metadata must carry nelec.")
    if nelec <= 0 or nelec % 2 != 0:
        raise QdynamicsConventionError("nelec", f"closed-shell export requires a positive even nelec; got {nelec}.")

    extraction = spatial_tensors_from_interleaved(elements.h1, elements.h2, atol=float(symmetry_atol))
    checks.update(extraction.checks)
    h_mo = extraction.h_mo
    eri_mo = extraction.eri_mo
    n = int(h_mo.shape[0])
    if nmo_declared is not None and nmo_declared != n:
        raise QdynamicsConventionError("nmo", f"metadata nmo={nmo_declared} but h1 has {n} spatial orbitals.")
    n_occ = nelec // 2
    if n_occ > n:
        raise QdynamicsConventionError("occupation", f"nelec/2={n_occ} exceeds {n} spatial orbitals.")
    if elements.mo_occ is not None:
        occ = np.asarray(elements.mo_occ, dtype=np.float64).reshape(-1)
        expected = np.concatenate([np.full(n_occ, 2.0), np.zeros(n - n_occ)])
        if occ.shape != expected.shape or np.max(np.abs(occ - expected)) > 0.0:
            raise QdynamicsConventionError("occupied_prefix", f"mo_occ={occ.tolist()} is not the doubly occupied prefix {expected.tolist()}.")
        checks["occupied_prefix_verified_from_mo_occ"] = True
    else:
        checks["occupied_prefix_verified_from_mo_occ"] = False
    nuclear = float(elements.nuclear_repulsion)
    witness_full = closed_shell_determinant_energy(nuclear, h_mo, eri_mo, n_occ)
    checks["input_tensor_determinant_energy_full_space"] = witness_full
    checks["hf_witness_abs_defect_full_space"] = abs(witness_full - e_scf)
    if abs(witness_full - e_scf) > float(witness_atol):
        raise QdynamicsConventionError(
            "hf_witness",
            f"E_det(tensors)={witness_full!r} disagrees with e_scf={e_scf!r} by {abs(witness_full - e_scf):.3e} > {witness_atol:.1e}.",
        )

    reduction: dict[str, Any] = {"kind": "none", "n_frozen": 0, "frozen_core_indices": [], "active_indices": list(range(n))}
    if frozen_core is not None and int(frozen_core.n_frozen) > 0:
        k = int(frozen_core.n_frozen)
        if k >= n_occ:
            raise QdynamicsConventionError("frozen_core", f"n_frozen={k} must be smaller than the occupied count {n_occ}.")
        frozen_idx = list(range(k))
        active_idx = list(range(k, n))
        core_scalar, h_active, g_active = frozen_core_contract(h_mo, eri_mo, nuclear, frozen=frozen_idx, active=active_idx)
        scalar_slot = core_scalar
        h_use, g_use = h_active, g_active
        n_use = n - k
        n_occ_use = n_occ - k
        reduction = {
            "kind": "frozen_core_occupied_prefix",
            "n_frozen": k,
            "frozen_core_indices": frozen_idx,
            "active_indices": active_idx,
            "frozen_core_scalar_hartree": float(core_scalar - nuclear),
            "formula": "E_core = E_nuc + 2 sum_c h_cc + sum_cd (2(cc|dd)-(cd|dc)); h'_pq = h_pq + sum_c (2(pq|cc)-(pc|cq))",
        }
        scalar_semantics = "nuclear_plus_frozen_core"
        core_policy = "already_contracted"
        frozen_scalar = float(core_scalar - nuclear)
    else:
        scalar_slot = nuclear
        h_use, g_use = h_mo, eri_mo
        n_use = n
        n_occ_use = n_occ
        scalar_semantics = "nuclear_only"
        core_policy = "none"
        frozen_scalar = 0.0
    witness_use = closed_shell_determinant_energy(scalar_slot, h_use, g_use, n_occ_use)
    checks["input_tensor_determinant_energy"] = witness_use
    checks["hf_witness_abs_defect"] = abs(witness_use - e_scf)
    if abs(witness_use - e_scf) > float(witness_atol):
        raise QdynamicsConventionError("frozen_core_witness", f"contracted determinant energy {witness_use!r} drifted from e_scf by {abs(witness_use - e_scf):.3e}.")

    unit = str(meta.get("unit", "")).strip()
    geometry_spec = str(meta.get("mol_str", "")).strip()
    problem = RestrictedClosedShellMolecularProblem(
        geometry_spec=geometry_spec,
        basis=str(meta.get("basis", "unknown")).strip(),
        charge=int(charge),
        multiplicity=1,
        reference="rhf",
        n_spatial_orbitals=int(n_use),
        n_alpha=int(n_occ_use),
        n_beta=int(n_occ_use),
        hf_energy=float(e_scf),
        nuclear_repulsion_energy=float(scalar_slot),
        one_body_integrals_mo=np.asarray(h_use, dtype=np.float64),
        two_body_integrals_mo=np.asarray(g_use, dtype=np.float64),
    )
    declaration = MolecularDiagnosticDeclaration(
        energy_units=INTAKE_ENERGY_UNITS,
        eri_convention=INTAKE_ERI_CONVENTION,
        orbital_basis=INTAKE_ORBITAL_BASIS,
        occupied_order=INTAKE_OCCUPIED_ORDER,
        scalar_semantics=scalar_semantics,
        nuclear_scalar_hartree=float(nuclear),
        frozen_core_scalar_hartree=float(frozen_scalar),
        one_body_core_policy=core_policy,
    )
    provenance = {
        "schema": BRIDGE_SCHEMA_VERSION,
        "schema_reference_producer": dict(PINNED_PRODUCER),
        "source": dict(elements.source),
        "origin": str(elements.source.get("origin", ORIGIN_UNKNOWN)),
        "container_schema": str(elements.source.get("container_schema", CONTAINER_IN_MEMORY)),
        "claimed_producer": elements.source.get("claimed_producer"),
        "origin_attestation": elements.source.get("origin_attestation"),
        "producer_origin_established": False,
        "metadata": {key: _decode_attr(value) for key, value in meta.items()},
        "geometry_units_as_exported": unit,
        "spin_orbital_order_as_exported": EXPECTED_H1_CONVENTION,
        "spin_orbital_order_consumed": "blocked_alpha_then_beta",
        "two_body_notation_declared": "chemists_(pq|rs)_real_spatial",
        "notation_binding": "declared_by_producer_metadata_not_inferred_from_symmetry",
        "scalar_as_exported": {"nuclear_repulsion": float(nuclear), "semantics": "nuclear_only"},
        "reduction": reduction,
        "full_space": {"n_spatial_orbitals": int(n), "n_alpha": int(n_occ), "n_beta": int(n_occ), "n_qubits": int(2 * n)},
        "consumed_space": {"n_spatial_orbitals": int(n_use), "n_alpha": int(n_occ_use), "n_beta": int(n_occ_use), "n_qubits": int(2 * n_use)},
        "spatial_tensor_sha256": {"h_mo_full": _array_sha256(h_mo), "eri_mo_full": _array_sha256(eri_mo), "h_consumed": _array_sha256(h_use), "g_consumed": _array_sha256(g_use)},
        "assumptions_not_verified_here": [
            "orbital orthonormality (canonical SCF orbitals in the producer's MO basis)",
            "energy units hartree (PySCF default)",
            "geometry units and bohr constant are the producer's",
        ],
    }
    return BridgeProblem(problem=problem, declaration=declaration, provenance=provenance, checks=checks)


# ----------------------------------------------------------------------------
# FCIDUMP (secondary path; PySCF ``tools.fcidump`` layout)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class FcidumpData:
    norb: int
    nelec: int
    ms2: int
    orbsym: tuple[int, ...]
    isym: int
    h: np.ndarray
    g: np.ndarray
    ecore: float
    unique_two_body_entries: int
    unique_one_body_entries: int
    text_sha256: str
    repeated_identical_entries: int = 0
    duplicate_atol: float = 0.0


_FCIDUMP_HEADER_INT = re.compile(r"(NORB|NELEC|MS2|ISYM)\s*=\s*([+-]?\d+)", re.IGNORECASE)
_FCIDUMP_ORBSYM = re.compile(r"ORBSYM\s*=\s*([0-9,\s]+)", re.IGNORECASE)


def _canonical_two_body_key(p: int, q: int, r: int, s: int) -> tuple[int, int, int, int]:
    """Representative of the eightfold real chemists' orbit of (pq|rs)."""

    pq = (max(p, q), min(p, q))
    rs = (max(r, s), min(r, s))
    first, second = (pq, rs) if pq >= rs else (rs, pq)
    return (first[0], first[1], second[0], second[1])


def _canonical_one_body_key(p: int, q: int) -> tuple[int, int]:
    return (max(p, q), min(p, q))


def read_fcidump(text: str, *, duplicate_atol: float = 0.0) -> FcidumpData:
    """Parse an FCIDUMP text in PySCF/Molpro layout (chemists' (ij|kl), 1-based).

    Lines ``value i j k l`` with ``k = l = 0`` are one-body entries, ``i = j = k
    = l = 0`` is the core/scalar energy, and the two-body entries are expanded
    with the eightfold real symmetry.  Every entry is keyed by its canonical
    symmetry representative: a repeated key with a value differing by more
    than ``duplicate_atol`` (default 0.0, exact repetition only) is a
    contradiction and is rejected; an identical repetition is counted in
    ``repeated_identical_entries``.  ``unique_*_entries`` count distinct
    canonical keys, not input lines.  ``[re, im]`` complex entries are rejected.
    """

    tolerance = float(duplicate_atol)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("duplicate_atol must be finite and non-negative")

    raw = str(text)
    header_end = None
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if re.search(r"&END|/", line.strip(), re.IGNORECASE) and index > 0:
            header_end = index
            break
    if header_end is None:
        raise QdynamicsConventionError("fcidump_header", "FCIDUMP header terminator (&END or /) not found.")
    header = "\n".join(lines[: header_end + 1])
    values = {match.group(1).upper(): int(match.group(2)) for match in _FCIDUMP_HEADER_INT.finditer(header)}
    for key in ("NORB", "NELEC"):
        if key not in values:
            raise QdynamicsConventionError("fcidump_header", f"FCIDUMP header lacks {key}.")
    norb = int(values["NORB"])
    nelec = int(values["NELEC"])
    ms2 = int(values.get("MS2", 0))
    isym = int(values.get("ISYM", 1))
    orbsym_match = _FCIDUMP_ORBSYM.search(header)
    orbsym = tuple(int(v) for v in re.findall(r"\d+", orbsym_match.group(1))) if orbsym_match else tuple([1] * norb)
    if norb <= 0:
        raise QdynamicsConventionError("fcidump_header", "NORB must be positive.")
    one_body: dict[tuple[int, int], float] = {}
    two_body: dict[tuple[int, int, int, int], float] = {}
    ecore = 0.0
    seen_core = False
    repeated = 0

    def _record(table: dict, key: tuple, value: float, line_text: str) -> None:
        nonlocal repeated
        if key in table:
            if abs(table[key] - value) > tolerance:
                raise QdynamicsConventionError(
                    "fcidump_conflict",
                    f"contradictory FCIDUMP entries for canonical key {key}: {table[key]!r} then {value!r} (line {line_text!r}; duplicate_atol={tolerance!r}).",
                )
            repeated += 1
            return
        table[key] = value
    for line in lines[header_end + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if "(" in stripped or "[" in stripped:
            raise QdynamicsConventionError("fcidump_complex", "complex FCIDUMP entries are not admissible.")
        parts = stripped.split()
        if len(parts) != 5:
            raise QdynamicsConventionError("fcidump_entry", f"malformed FCIDUMP line: {stripped!r}.")
        try:
            value = float(parts[0].replace("D", "E").replace("d", "e"))
            i, j, k, l = (int(p) for p in parts[1:])
        except ValueError as exc:
            raise QdynamicsConventionError("fcidump_entry", f"malformed FCIDUMP line: {stripped!r}.") from exc
        if not math.isfinite(value):
            raise QdynamicsConventionError("fcidump_nonfinite", f"nonfinite FCIDUMP entry: {stripped!r}.")
        if i == j == k == l == 0:
            if seen_core:
                raise QdynamicsConventionError("fcidump_entry", "duplicate core-energy entry.")
            ecore = value
            seen_core = True
            continue
        if k == 0 and l == 0:
            if not (1 <= i <= norb and 1 <= j <= norb):
                raise QdynamicsConventionError("fcidump_index", f"one-body index out of range: {stripped!r}.")
            _record(one_body, _canonical_one_body_key(i - 1, j - 1), value, stripped)
            continue
        if not all(1 <= idx <= norb for idx in (i, j, k, l)):
            raise QdynamicsConventionError("fcidump_index", f"two-body index out of range: {stripped!r}.")
        _record(two_body, _canonical_two_body_key(i - 1, j - 1, k - 1, l - 1), value, stripped)
    h = np.zeros((norb, norb), dtype=np.float64)
    g = np.zeros((norb, norb, norb, norb), dtype=np.float64)
    for (p, q), value in one_body.items():
        h[p, q] = value
        h[q, p] = value
    for (p, q, r, s), value in two_body.items():
        for a, b, c, d in ((p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r), (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p)):
            g[a, b, c, d] = value
    unique_two = len(two_body)
    unique_one = len(one_body)
    return FcidumpData(
        norb=norb,
        nelec=nelec,
        ms2=ms2,
        orbsym=orbsym,
        isym=isym,
        h=h,
        g=g,
        ecore=float(ecore),
        unique_two_body_entries=int(unique_two),
        unique_one_body_entries=int(unique_one),
        text_sha256=_sha256_bytes(raw.encode("utf-8")),
        repeated_identical_entries=int(repeated),
        duplicate_atol=tolerance,
    )


@dataclass(frozen=True)
class FcidumpSidecar:
    """Declarations an FCIDUMP cannot carry; every field is bound by the producer, not inferred."""

    geometry_spec: str
    basis: str
    e_scf: float
    integral_notation: str = "chemists_spatial_real"
    scalar_semantics: str = "nuclear_only"
    nuclear_scalar_hartree: float | None = None
    frozen_core_scalar_hartree: float = 0.0
    charge: int = 0
    units: str = "angstrom"
    producer: Mapping[str, Any] = field(default_factory=dict)
    origin: str = ORIGIN_UNKNOWN
    attestation: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if str(self.origin) not in ORIGIN_CHOICES:
            raise ValueError(f"origin must be one of {ORIGIN_CHOICES}; got {self.origin!r}")
        if str(self.origin) == ORIGIN_ATTESTED:
            if self.attestation is None:
                raise ValueError("origin='attested' requires attestation={'attested_by': <who>, 'text_sha256': <sha256 of the FCIDUMP text>, ...}")
            try:
                validate_attestation(self.attestation, digest_key="text_sha256")
            except QdynamicsConventionError as exc:
                raise ValueError(f"origin='attested' requires a valid attestation: {exc}") from exc
        elif self.attestation:
            raise ValueError(f"attestation supplied but origin is {self.origin!r}; declare origin='attested' explicitly")


def bridge_problem_from_fcidump(
    data: FcidumpData,
    sidecar: FcidumpSidecar,
    *,
    symmetry_atol: float = 1.0e-12,
    witness_atol: float = 1.0e-8,
) -> BridgeProblem:
    if str(sidecar.integral_notation) != "chemists_spatial_real":
        raise QdynamicsConventionError("fcidump_notation", f"only chemists_spatial_real FCIDUMP is admissible; declared {sidecar.integral_notation!r}.")
    if int(data.ms2) != 0:
        raise QdynamicsConventionError("open_shell", f"MS2={data.ms2}; closed shell required (MS2=0 is necessary, not sufficient).")
    if int(data.nelec) <= 0 or int(data.nelec) % 2 != 0:
        raise QdynamicsConventionError("nelec", f"closed-shell FCIDUMP requires even NELEC; got {data.nelec}.")
    n_occ = int(data.nelec) // 2
    if n_occ > int(data.norb):
        raise QdynamicsConventionError("occupation", "NELEC/2 exceeds NORB.")
    checks: dict[str, Any] = {
        "h_hermitian_max_abs": float(np.max(np.abs(data.h - data.h.T))),
        "eri_first_pair_max_abs": float(np.max(np.abs(data.g - data.g.transpose(1, 0, 2, 3)))),
        "eri_second_pair_max_abs": float(np.max(np.abs(data.g - data.g.transpose(0, 1, 3, 2)))),
        "eri_pair_exchange_max_abs": float(np.max(np.abs(data.g - data.g.transpose(2, 3, 0, 1)))),
    }
    if any(value > float(symmetry_atol) for value in checks.values()):
        raise QdynamicsConventionError("eri_symmetry", f"FCIDUMP tensors violate real chemists' symmetry: {checks}.")
    semantics = str(sidecar.scalar_semantics)
    if semantics == "nuclear_only":
        nuclear = float(data.ecore) if sidecar.nuclear_scalar_hartree is None else float(sidecar.nuclear_scalar_hartree)
        if abs(nuclear - float(data.ecore)) > float(symmetry_atol):
            raise QdynamicsConventionError("scalar_declaration", "declared nuclear scalar differs from the FCIDUMP core energy under nuclear_only semantics.")
        frozen_scalar = 0.0
        core_policy = "none"
    elif semantics == "nuclear_plus_frozen_core":
        if sidecar.nuclear_scalar_hartree is None:
            raise QdynamicsConventionError("scalar_declaration", "nuclear_plus_frozen_core requires the nuclear scalar separately.")
        nuclear = float(sidecar.nuclear_scalar_hartree)
        frozen_scalar = float(sidecar.frozen_core_scalar_hartree)
        if abs(nuclear + frozen_scalar - float(data.ecore)) > float(symmetry_atol):
            raise QdynamicsConventionError("scalar_declaration", "nuclear + frozen-core scalars do not sum to the FCIDUMP core energy.")
        core_policy = "already_contracted"
    else:
        raise QdynamicsConventionError("scalar_declaration", f"unsupported scalar_semantics {semantics!r}.")
    witness = closed_shell_determinant_energy(float(data.ecore), data.h, data.g, n_occ)
    checks["input_tensor_determinant_energy"] = witness
    checks["hf_witness_abs_defect"] = abs(witness - float(sidecar.e_scf))
    if abs(witness - float(sidecar.e_scf)) > float(witness_atol):
        raise QdynamicsConventionError("hf_witness", f"E_det={witness!r} disagrees with declared e_scf={sidecar.e_scf!r} by {abs(witness - float(sidecar.e_scf)):.3e}.")
    attestation_record: dict[str, Any] | None = None
    if str(sidecar.origin) == ORIGIN_ATTESTED:
        if str(sidecar.attestation["text_sha256"]).lower() != str(data.text_sha256).lower():  # type: ignore[index]
            raise QdynamicsConventionError("origin_attestation", "attestation.text_sha256 does not match the parsed FCIDUMP text.")
        attestation_record = {str(k): v for k, v in sidecar.attestation.items()}  # type: ignore[union-attr]
    problem = RestrictedClosedShellMolecularProblem(
        geometry_spec=str(sidecar.geometry_spec),
        basis=str(sidecar.basis),
        charge=int(sidecar.charge),
        multiplicity=1,
        reference="rhf",
        n_spatial_orbitals=int(data.norb),
        n_alpha=n_occ,
        n_beta=n_occ,
        hf_energy=float(sidecar.e_scf),
        nuclear_repulsion_energy=float(data.ecore),
        one_body_integrals_mo=np.asarray(data.h, dtype=np.float64),
        two_body_integrals_mo=np.asarray(data.g, dtype=np.float64),
    )
    declaration = MolecularDiagnosticDeclaration(
        energy_units=INTAKE_ENERGY_UNITS,
        eri_convention=INTAKE_ERI_CONVENTION,
        orbital_basis=INTAKE_ORBITAL_BASIS,
        occupied_order=INTAKE_OCCUPIED_ORDER,
        scalar_semantics=semantics,
        nuclear_scalar_hartree=float(nuclear),
        frozen_core_scalar_hartree=float(frozen_scalar),
        one_body_core_policy=core_policy,
    )
    provenance = {
        "schema": BRIDGE_SCHEMA_VERSION,
        "source": {"container_schema": "fcidump_text", "text_sha256": data.text_sha256, "unique_two_body_entries": data.unique_two_body_entries, "unique_one_body_entries": data.unique_one_body_entries, "repeated_identical_entries": data.repeated_identical_entries, "duplicate_atol": data.duplicate_atol, "orbsym": list(data.orbsym), "isym": data.isym, "origin": str(sidecar.origin)},
        "schema_reference_producer": dict(PINNED_PRODUCER),
        "claimed_producer": (dict(sidecar.producer).get("claimed_producer") if sidecar.producer else None),
        "origin": str(sidecar.origin),
        "container_schema": "fcidump_text",
        "origin_attestation": attestation_record,
        "producer_origin_established": False,
        "notation_binding": "declared_by_sidecar_not_inferred_from_symmetry",
        "geometry_units_as_declared": str(sidecar.units),
        "reduction": {"kind": "none_declared_by_sidecar", "scalar_semantics": semantics},
        "consumed_space": {"n_spatial_orbitals": int(data.norb), "n_alpha": n_occ, "n_beta": n_occ, "n_qubits": int(2 * data.norb)},
        "fcidump_is_current_producer_output": False,
    }
    return BridgeProblem(problem=problem, declaration=declaration, provenance=provenance, checks=checks)


# ----------------------------------------------------------------------------
# Serialization for the strict intake
# ----------------------------------------------------------------------------


def bridge_problem_to_jsonable(bridge_problem: BridgeProblem) -> dict[str, Any]:
    """Document consumed by the strict intake at payload location ``$.problem``."""

    return {
        "schema": BRIDGE_PROBLEM_SCHEMA,
        "problem": bridge_problem.problem.to_jsonable(),
        "declaration": asdict(bridge_problem.declaration),
        "provenance": bridge_problem.provenance,
        "checks": bridge_problem.checks,
    }


def _json_dumps_strict(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=False) + "\n"


def write_bridge_problem_json(bridge_problem: BridgeProblem, path: str | Path) -> dict[str, Any]:
    """Write the problem document and a sidecar manifest; return the manifest.

    The manifest records the document's exact ``raw_sha256`` and the declaration
    so a later ``resolve_bridge_context`` can call the strict intake with the
    expected digest.  Editing the document afterwards fails that intake.
    """

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_dumps_strict(bridge_problem_to_jsonable(bridge_problem)).encode("utf-8")
    out_path.write_bytes(raw)
    manifest = {
        "schema": BRIDGE_MANIFEST_SCHEMA,
        "problem_json": out_path.name,
        "raw_sha256": _sha256_bytes(raw),
        "raw_size_bytes": len(raw),
        "declaration": asdict(bridge_problem.declaration),
        "n_spatial_orbitals": int(bridge_problem.problem.n_spatial_orbitals),
        "n_qubits": int(2 * bridge_problem.problem.n_spatial_orbitals),
        "num_particles": [int(bridge_problem.problem.n_alpha), int(bridge_problem.problem.n_beta)],
        "provenance_source": dict(bridge_problem.provenance.get("source", {})),
        "reduction": dict(bridge_problem.provenance.get("reduction", {})),
    }
    manifest_path = out_path.with_name(out_path.stem + ".manifest.json")
    manifest_path.write_bytes(_json_dumps_strict(manifest).encode("utf-8"))
    manifest["manifest_path"] = str(manifest_path)
    manifest["problem_json_path"] = str(out_path)
    return manifest


def read_bridge_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != BRIDGE_MANIFEST_SCHEMA:
        raise ValueError(f"not a bridge manifest: {manifest_path}")
    payload["manifest_path"] = str(manifest_path)
    payload["problem_json_path"] = str(manifest_path.with_name(str(payload["problem_json"])))
    return payload


def declaration_from_mapping(payload: Mapping[str, Any]) -> MolecularDiagnosticDeclaration:
    return MolecularDiagnosticDeclaration(
        energy_units=str(payload["energy_units"]),
        eri_convention=str(payload["eri_convention"]),
        orbital_basis=str(payload["orbital_basis"]),
        occupied_order=str(payload["occupied_order"]),
        scalar_semantics=str(payload["scalar_semantics"]),
        nuclear_scalar_hartree=float(payload["nuclear_scalar_hartree"]),
        frozen_core_scalar_hartree=float(payload["frozen_core_scalar_hartree"]),
        one_body_core_policy=str(payload["one_body_core_policy"]),
    )


__all__ = [
    "BRIDGE_MANIFEST_SCHEMA",
    "BRIDGE_PROBLEM_SCHEMA",
    "BridgeDependencyUnavailable",
    "BridgeProblem",
    "CONTAINER_IN_MEMORY",
    "ORIGIN_ATTESTED",
    "ORIGIN_CHOICES",
    "ORIGIN_SIMULATED",
    "ORIGIN_SYNTHETIC",
    "ORIGIN_UNKNOWN",
    "EXPECTED_H1_CONVENTION",
    "EXPECTED_H2_CONVENTION",
    "FcidumpData",
    "FcidumpSidecar",
    "FrozenCoreReduction",
    "PINNED_PRODUCER",
    "QDYNAMICS_ELE_H5_SCHEMA",
    "QdynamicsConventionError",
    "QdynamicsMatrixElements",
    "bridge_problem_from_fcidump",
    "bridge_problem_from_qdynamics",
    "bridge_problem_to_jsonable",
    "closed_shell_determinant_energy",
    "declaration_from_mapping",
    "frozen_core_contract",
    "load_qdynamics_matrix_elements_h5",
    "qdynamics_matrix_elements_from_mapping",
    "read_bridge_manifest",
    "read_fcidump",
    "spatial_tensors_from_interleaved",
    "validate_attestation",
    "write_bridge_problem_json",
]
