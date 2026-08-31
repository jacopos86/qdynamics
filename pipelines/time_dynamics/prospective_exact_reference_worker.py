"""Source-locked exact-reference worker for prospective Paper-II cells.

The bootstrap planner deliberately launches an exact-reference action beside
the exact-blind AVQDS scout after a seed qualifies.  This module owns the
exact-reference side of that fork.  It authenticates the planner request,
source epoch, qualified seed receipt, and seed bytes before any numerical
work, then atomically writes the data, manifest, and typed completion receipt.

The implementation is intentionally local to one action.  It does not contact
Condor, mutate the acquisition registry, score a variational trajectory, or
enable production submission.  Replaying the same work order is idempotent;
different bytes at an already materialized content address fail closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.scaffold.hh_fixed_manifold_loader import (
    _statevector_from_named_payload_state,
)
from pipelines.time_dynamics.adapters.observables import hh_observable_snapshot
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    ExactReferenceIdentity,
    LANE_HH,
    LANE_HUBBARD,
    SeedIdentity,
    SourceEpochIdentity,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    EXACT_REFERENCE_DATA_V1,
    EXACT_REFERENCE_MANIFEST_V1,
    EXACT_REFERENCE_RECEIPT_V1,
    EXEC_HH_REFERENCE,
    EXEC_HUBBARD_REFERENCE,
    ExactReferenceReceipt,
    ExecutionRequest,
    ProspectiveExecutionError,
    SeedQualificationReceipt,
    SourceEpochReceipt,
    executor_bindings,
    resolve_bootstrap_action,
)
from src.quantum.drives_time_potential import (
    default_spatial_weights,
    gaussian_sinusoid_waveform,
    reference_method_name,
)
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
    mode_index,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


WORK_ORDER_SCHEMA_V1 = "paper_ii_exact_reference_work_order_v1"
PROPAGATION_POLICY_SCHEMA_V1 = "paper_ii_exact_reference_propagation_policy_v1"
EXACT_REFERENCE_WORKER_SCHEMA_V1 = "paper_ii_exact_reference_worker_v1"

_WORKER_SOURCE_PATH = "pipelines/time_dynamics/prospective_exact_reference_worker.py"
_RUNNER_SOURCE_PATH = (
    "pipelines/time_dynamics/runners/build_prospective_exact_reference.py"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pretty_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _seal(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(payload)
    if key in result:
        raise ProspectiveExecutionError(f"payload already contains {key}")
    result[key] = _canonical_sha(result)
    return result


def _verify_seal(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    core = dict(payload)
    declared = core.pop(key, None)
    actual = _canonical_sha(core)
    if declared != actual:
        raise ProspectiveExecutionError(f"{label} self digest mismatch")
    return actual


def _atomic_write_immutable(path: Path, content: bytes) -> None:
    """Install bytes atomically, accepting only byte-identical crash replay."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ProspectiveExecutionError(
                f"refusing to overwrite immutable exact-reference evidence: {path}"
            )
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _source_epoch_from_request(request: ExecutionRequest) -> SourceEpochIdentity:
    raw = request.inputs.get("source_epoch")
    if not isinstance(raw, Mapping):
        raise ProspectiveExecutionError("exact-reference request has no source epoch")
    try:
        epoch = SourceEpochIdentity(
            lane_id=str(raw.get("lane_id") or ""),
            source_epoch_id=str(raw.get("source_epoch_id") or ""),
            source_revision=raw.get("source_revision"),
            source_archive_sha256=raw.get("source_archive_sha256"),
        )
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "exact-reference source epoch is invalid"
        ) from exc
    if raw.get("source_epoch_identity_sha256") != epoch.identity_sha256:
        raise ProspectiveExecutionError("exact-reference source epoch digest mismatch")
    if epoch.lane_id != request.lane_id:
        raise ProspectiveExecutionError("exact-reference source epoch crosses lanes")
    return epoch


def _canonical_reference_and_seed(
    request: ExecutionRequest,
) -> tuple[ExactReferenceIdentity, SeedIdentity, SourceEpochIdentity]:
    expected_binding = {
        LANE_HH: EXEC_HH_REFERENCE,
        LANE_HUBBARD: EXEC_HUBBARD_REFERENCE,
    }.get(request.lane_id)
    if (
        expected_binding is None
        or request.kind != "build_exact_reference"
        or request.binding_id != expected_binding
        or request.exact_reference_blind
        or request.attempt != 0
        or request.planner_sequence != 0
        or request.requested_memory_gib is not None
        or request.arm is not None
        or request.inputs.get("reporting_only") is not True
        or request.inputs.get("exact_data_schema") != EXACT_REFERENCE_DATA_V1
    ):
        raise ProspectiveExecutionError(
            "request is not a canonical prospective exact-reference action"
        )
    epoch = _source_epoch_from_request(request)
    registry = build_identity_registry()
    references = tuple(
        row
        for row in registry.references
        if row.reference_id == request.inputs.get("reference_id")
    )
    if len(references) != 1:
        raise ProspectiveExecutionError(
            "request does not name one canonical exact reference"
        )
    reference = references[0]
    seeds = tuple(row for row in registry.seeds if row.seed_id == reference.seed_id)
    if len(seeds) != 1:
        raise ProspectiveExecutionError("exact reference has no canonical seed")
    seed = seeds[0]
    if reference.lane_id != request.lane_id or seed.lane_id != request.lane_id:
        raise ProspectiveExecutionError("exact-reference request crosses lanes")
    seed_sha = str(request.inputs.get("seed_artifact_sha256") or "")
    plan = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(
            seed_artifact_sha256={seed.seed_id: seed_sha},
            source_epochs={request.lane_id: epoch},
        ),
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    actions = tuple(
        row for row in plan.ready_actions if row.action_id == request.action_id
    )
    if len(actions) != 1:
        raise ProspectiveExecutionError(
            "exact-reference request is not emitted by the canonical planner"
        )
    canonical = resolve_bootstrap_action(actions[0])
    if canonical.to_json_dict() != request.to_json_dict():
        raise ProspectiveExecutionError(
            "exact-reference request diverges from canonical inputs"
        )
    if (
        reference.propagator != "paper_ii_exact_driven_reference_v1"
        or reference.model not in {"hubbard_holstein", "hubbard"}
    ):
        raise ProspectiveExecutionError(
            "exact-reference request changed the frozen model/propagator"
        )
    return reference, seed, epoch


def propagation_policy(reference: ExactReferenceIdentity) -> dict[str, Any]:
    """Return the deterministic implementation of the frozen propagator ID."""

    drive = dict(reference.drive)
    if (
        drive.get("operator") != "staggered_site_density"
        or drive.get("pattern") != "staggered"
        or drive.get("time_sampling") != "midpoint"
        or drive.get("include_identity") is not False
        or float(drive.get("tbar", math.nan)) != 1.0
        or float(drive.get("phi", math.nan)) != 0.0
        or float(drive.get("t0", math.nan)) != 0.0
    ):
        raise ProspectiveExecutionError(
            "exact-reference drive diverges from the frozen waveform"
        )
    worker_source = Path(__file__).resolve(strict=True)
    runner_source = worker_source.parent / "runners" / (
        "build_prospective_exact_reference.py"
    )
    if not runner_source.is_file():
        raise ProspectiveExecutionError(
            "exact-reference runner source is missing from the executor overlay"
        )
    executor_sources = {
        _WORKER_SOURCE_PATH: _file_sha(worker_source),
        _RUNNER_SOURCE_PATH: _file_sha(runner_source),
    }
    payload = {
        "schema": PROPAGATION_POLICY_SCHEMA_V1,
        "propagator": reference.propagator,
        "implementation": reference_method_name("midpoint"),
        "time_sampling": "midpoint",
        "microsteps_per_ordered_grid_interval": 1,
        "energy_sampling": "instantaneous_hamiltonian_at_checkpoint",
        "state_normalization_check_abs_tol": 1.0e-10,
        "source_composition": "source_epoch_plus_digest_bound_executor_overlay_v1",
        "executor_source_files": executor_sources,
        "executor_source_files_sha256": _canonical_sha(executor_sources),
        "qiskit_compilation": False,
    }
    payload["propagation_policy_sha256"] = _canonical_sha(payload)
    return payload


@dataclass(frozen=True)
class ExactReferenceWorkOrder:
    request: ExecutionRequest
    reference: ExactReferenceIdentity
    seed: SeedIdentity
    source_epoch_receipt_path: str
    source_epoch_receipt_sha256: str
    seed_receipt_path: str
    seed_receipt_sha256: str
    seed_artifact_path: str
    seed_artifact_sha256: str
    propagation: Mapping[str, Any]
    work_order_sha256: str

    def to_json_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema": WORK_ORDER_SCHEMA_V1,
            "execution_request": self.request.to_json_dict(),
            "reference_identity": self.reference.to_json_dict(),
            "seed_identity": self.seed.to_json_dict(),
            "source_epoch_receipt": {
                "path": self.source_epoch_receipt_path,
                "sha256": self.source_epoch_receipt_sha256,
            },
            "seed_receipt": {
                "path": self.seed_receipt_path,
                "sha256": self.seed_receipt_sha256,
            },
            "seed_artifact": {
                "path": self.seed_artifact_path,
                "sha256": self.seed_artifact_sha256,
            },
            "propagation_policy": dict(self.propagation),
            "remote_raw_evidence_policy": "preserve",
            "production_submission_enabled": False,
        }
        if include_digest:
            payload["work_order_sha256"] = self.work_order_sha256
        return payload


def build_work_order(
    request: ExecutionRequest,
    *,
    source_epoch_receipt: SourceEpochReceipt,
    seed_receipt: SeedQualificationReceipt,
) -> ExactReferenceWorkOrder:
    reference, seed, epoch = _canonical_reference_and_seed(request)
    if (
        source_epoch_receipt.source_epoch != epoch
        or source_epoch_receipt.lane_id != request.lane_id
    ):
        raise ProspectiveExecutionError(
            "exact-reference source receipt/request mismatch"
        )
    if (
        seed_receipt.seed_id != seed.seed_id
        or seed_receipt.seed_identity_sha256 != seed.identity_sha256
        or seed_receipt.source_epoch_identity_sha256 != epoch.identity_sha256
        or seed_receipt.artifact.sha256
        != request.inputs.get("seed_artifact_sha256")
        or reference.materialized_key(seed_receipt.artifact.sha256)
        != request.inputs.get("materialized_reference_key")
    ):
        raise ProspectiveExecutionError(
            "exact-reference qualified seed binding mismatch"
        )
    policy = propagation_policy(reference)
    core = {
        "schema": WORK_ORDER_SCHEMA_V1,
        "execution_request": request.to_json_dict(),
        "reference_identity": reference.to_json_dict(),
        "seed_identity": seed.to_json_dict(),
        "source_epoch_receipt": {
            "path": source_epoch_receipt.receipt_path,
            "sha256": source_epoch_receipt.receipt_sha256,
        },
        "seed_receipt": {
            "path": seed_receipt.receipt_path,
            "sha256": seed_receipt.receipt_sha256,
        },
        "seed_artifact": seed_receipt.artifact.to_json_dict(),
        "propagation_policy": policy,
        "remote_raw_evidence_policy": "preserve",
        "production_submission_enabled": False,
    }
    return ExactReferenceWorkOrder(
        request=request,
        reference=reference,
        seed=seed,
        source_epoch_receipt_path=source_epoch_receipt.receipt_path,
        source_epoch_receipt_sha256=source_epoch_receipt.receipt_sha256,
        seed_receipt_path=seed_receipt.receipt_path,
        seed_receipt_sha256=seed_receipt.receipt_sha256,
        seed_artifact_path=seed_receipt.artifact.path,
        seed_artifact_sha256=seed_receipt.artifact.sha256,
        propagation=policy,
        work_order_sha256=_canonical_sha(core),
    )


def write_work_order(path: str | Path, order: ExactReferenceWorkOrder) -> Path:
    target = Path(path)
    _atomic_write_immutable(target, _pretty_bytes(order.to_json_dict()))
    return target


def _repository_file(
    root: Path, path: str, sha256: str, *, label: str
) -> Path:
    candidate = (root / path).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError(f"{label} escapes repository root") from exc
    if _file_sha(candidate) != sha256:
        raise ProspectiveExecutionError(f"{label} file digest mismatch")
    return candidate


def load_work_order(
    path: str | Path, *, repository_root: str | Path
) -> tuple[ExactReferenceWorkOrder, SourceEpochReceipt, SeedQualificationReceipt]:
    root = Path(repository_root).resolve()
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "exact-reference work order is outside repository"
        ) from exc
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError(
            "exact-reference work order is unreadable"
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != WORK_ORDER_SCHEMA_V1:
        raise ProspectiveExecutionError("unsupported exact-reference work order")
    work_sha = _verify_seal(
        payload, "work_order_sha256", label="exact-reference work order"
    )
    request_row = payload.get("execution_request")
    source_row = payload.get("source_epoch_receipt")
    seed_row = payload.get("seed_receipt")
    artifact_row = payload.get("seed_artifact")
    if not all(
        isinstance(value, Mapping)
        for value in (request_row, source_row, seed_row, artifact_row)
    ):
        raise ProspectiveExecutionError("exact-reference work order is incomplete")
    request = ExecutionRequest.from_mapping(request_row)
    reference, seed, epoch = _canonical_reference_and_seed(request)
    source_path = _repository_file(
        root,
        str(source_row.get("path") or ""),
        str(source_row.get("sha256") or ""),
        label="source epoch receipt",
    )
    source_receipt = SourceEpochReceipt.load(
        source_path, repository_root=root, expected_epoch=epoch
    )
    seed_path = _repository_file(
        root,
        str(seed_row.get("path") or ""),
        str(seed_row.get("sha256") or ""),
        label="seed qualification receipt",
    )
    seed_receipt = SeedQualificationReceipt.load(
        seed_path,
        repository_root=root,
        expected_seed=seed,
        expected_source_epoch=epoch,
    )
    if artifact_row != seed_receipt.artifact.to_json_dict():
        raise ProspectiveExecutionError(
            "exact-reference work order seed artifact binding diverged"
        )
    rebuilt = build_work_order(
        request,
        source_epoch_receipt=source_receipt,
        seed_receipt=seed_receipt,
    )
    if (
        payload.get("reference_identity") != reference.to_json_dict()
        or payload.get("seed_identity") != seed.to_json_dict()
        or payload.get("propagation_policy") != propagation_policy(reference)
        or payload.get("remote_raw_evidence_policy") != "preserve"
        or payload.get("production_submission_enabled") is not False
        or rebuilt.work_order_sha256 != work_sha
    ):
        raise ProspectiveExecutionError(
            "exact-reference work order canonical contract changed"
        )
    return rebuilt, source_receipt, seed_receipt


@dataclass(frozen=True)
class ExactReferenceComputation:
    time: Sequence[float]
    energy: Sequence[float]
    doublon: Sequence[float]
    site_occupations: Sequence[Sequence[float]]
    states: Sequence[Sequence[complex]]


def _static_hamiltonian(reference: ExactReferenceIdentity) -> np.ndarray:
    physical = dict(reference.physical)
    if reference.lane_id == LANE_HH:
        polynomial = build_hubbard_holstein_hamiltonian(
            dims=int(physical["lattice_sites"]),
            J=float(physical["t_hop"]),
            U=float(physical["U"]),
            omega0=float(physical["omega0"]),
            g=float(physical["g"]),
            n_ph_max=int(physical["n_ph_max"]),
            boson_encoding=str(physical["boson_encoding"]),
            v0=float(physical["staggered_potential"]),
            repr_mode="JW",
            indexing=str(physical["ordering"]),
            pbc=str(physical["boundary"]).strip().lower() == "periodic",
            include_zero_point=True,
        )
    elif reference.lane_id == LANE_HUBBARD:
        hopping = float(physical["t_hop"])
        polynomial = build_hubbard_hamiltonian(
            dims=int(physical["lattice_sites"]),
            t=hopping,
            U=float(physical["U_over_t"]) * hopping,
            v=0.0,
            repr_mode="JW",
            indexing=str(physical["ordering"]),
            pbc=str(physical["boundary"]).strip().lower() == "periodic",
        )
    else:  # pragma: no cover - canonical validator rejects this first
        raise ProspectiveExecutionError("unsupported exact-reference lane")
    matrix = np.asarray(hamiltonian_matrix(polynomial), dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ProspectiveExecutionError("exact-reference Hamiltonian is not square")
    if not np.allclose(matrix, np.conjugate(matrix.T), rtol=0.0, atol=1.0e-11):
        raise ProspectiveExecutionError("exact-reference Hamiltonian is not Hermitian")
    return matrix


def _drive_diagonal(reference: ExactReferenceIdentity, dimension: int) -> np.ndarray:
    physical = dict(reference.physical)
    n_sites = int(physical["lattice_sites"])
    ordering = str(physical["ordering"])
    weights = default_spatial_weights(n_sites, mode="staggered")
    diagonal = np.zeros(int(dimension), dtype=float)
    for basis_index in range(int(dimension)):
        value = 0.0
        for site in range(n_sites):
            for spin in (0, 1):
                qubit = mode_index(site, spin, indexing=ordering, n_sites=n_sites)
                value += float(weights[site]) * float(
                    (basis_index >> int(qubit)) & 1
                )
        diagonal[basis_index] = value
    return diagonal


def _waveform(reference: ExactReferenceIdentity, time_value: float) -> float:
    drive = reference.drive
    return gaussian_sinusoid_waveform(
        float(time_value) + float(drive["t0"]),
        A=float(drive["amplitude"]),
        omega=float(drive["omega"]),
        tbar=float(drive["tbar"]),
        phi=float(drive["phi"]),
    )


def compute_exact_reference(order: ExactReferenceWorkOrder) -> ExactReferenceComputation:
    """Run the frozen exponential-midpoint reference for one cell."""

    try:
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import expm_multiply
    except ImportError as exc:  # pragma: no cover - runtime dependency contract
        raise ProspectiveExecutionError(
            "SciPy is required for the exact-reference worker"
        ) from exc

    seed_path = Path(order.seed_artifact_path)
    if not seed_path.is_file():
        raise ProspectiveExecutionError(
            "default exact-reference compute requires repository-root cwd"
        )
    try:
        seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveExecutionError("qualified seed artifact is unreadable") from exc
    if not isinstance(seed_payload, Mapping):
        raise ProspectiveExecutionError("qualified seed artifact must be an object")
    psi = np.asarray(
        _statevector_from_named_payload_state(seed_payload, "initial_state"),
        dtype=complex,
    ).reshape(-1)
    static = _static_hamiltonian(order.reference)
    if static.shape != (psi.size, psi.size):
        raise ProspectiveExecutionError(
            "qualified seed and exact-reference Hamiltonian dimensions differ"
        )
    drive_diagonal = _drive_diagonal(order.reference, psi.size)
    diagonal_indices = np.diag_indices_from(static)
    grid = order.reference.grid
    points = int(grid["points"])
    start = float(grid["start"])
    step = float(grid["step"])
    times = tuple(start + float(index) * step for index in range(points))
    if _canonical_sha(times) != grid.get("ordered_times_sha256"):
        raise ProspectiveExecutionError("exact-reference ordered grid changed")

    states: list[np.ndarray] = []
    energies: list[float] = []
    doublons: list[float] = []
    occupations: list[tuple[float, ...]] = []
    for index, time_value in enumerate(times):
        norm = float(np.linalg.norm(psi))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1.0e-10:
            raise ProspectiveExecutionError(
                "exact-reference propagator returned a non-normalized state"
            )
        psi = np.asarray(psi / norm, dtype=complex)
        instantaneous = np.array(static, copy=True)
        instantaneous[diagonal_indices] += _waveform(order.reference, time_value) * drive_diagonal
        energy = float(np.real(np.vdot(psi, instantaneous @ psi)))
        if not math.isfinite(energy):
            raise ProspectiveExecutionError("exact-reference energy is nonfinite")
        snapshot = hh_observable_snapshot(
            psi,
            num_sites=int(order.reference.physical["lattice_sites"]),
            ordering=str(order.reference.physical["ordering"]),
        )
        states.append(np.array(psi, copy=True))
        energies.append(energy)
        doublons.append(float(snapshot["doublon"]))
        occupations.append(tuple(float(value) for value in snapshot["site_occupations"]))
        if index + 1 < len(times):
            midpoint = 0.5 * (float(time_value) + float(times[index + 1]))
            step_hamiltonian = np.array(static, copy=True)
            step_hamiltonian[diagonal_indices] += _waveform(order.reference, midpoint) * drive_diagonal
            generator = csc_matrix((-1.0j * step) * step_hamiltonian)
            psi = np.asarray(expm_multiply(generator, psi), dtype=complex).reshape(-1)
    return ExactReferenceComputation(
        time=times,
        energy=tuple(energies),
        doublon=tuple(doublons),
        site_occupations=tuple(occupations),
        states=tuple(tuple(complex(value) for value in state) for state in states),
    )


def _reference_contract(
    reference: ExactReferenceIdentity, seed: SeedIdentity
) -> dict[str, Any]:
    return {
        "reference_id": reference.reference_id,
        "reference_identity_sha256": reference.identity_sha256,
        "lane_id": reference.lane_id,
        "cell_id": reference.cell_id,
        "seed_id": reference.seed_id,
        "seed_identity_sha256": seed.identity_sha256,
        "model": reference.model,
        "physical": dict(reference.physical),
        "drive": dict(reference.drive),
        "grid": dict(reference.grid),
        "propagator": reference.propagator,
        "observable_schema": reference.observable_schema,
    }


def _normalize_computation(
    result: ExactReferenceComputation, *, order: ExactReferenceWorkOrder
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[complex, ...], ...],
]:
    try:
        times = tuple(float(value) for value in result.time)
        energy = tuple(float(value) for value in result.energy)
        doublon = tuple(float(value) for value in result.doublon)
        occupations = tuple(
            tuple(float(value) for value in row) for row in result.site_occupations
        )
        states = tuple(
            tuple(complex(value) for value in row) for row in result.states
        )
    except (TypeError, ValueError) as exc:
        raise ProspectiveExecutionError(
            "exact-reference computation arrays are malformed"
        ) from exc
    points = int(order.reference.grid["points"])
    sites = int(order.reference.physical["lattice_sites"])
    if order.reference.lane_id == LANE_HUBBARD:
        expected_state_width = 1 << (2 * sites)
    elif order.reference.lane_id == LANE_HH:
        levels = int(order.reference.physical["n_ph_max"]) + 1
        phonon_qubits = max(1, (levels - 1).bit_length())
        expected_state_width = 1 << (2 * sites + sites * phonon_qubits)
    else:  # pragma: no cover - canonical work-order validation rejects this
        raise ProspectiveExecutionError(
            "exact-reference computation has an unsupported canonical model"
        )
    if (
        any(len(row) != sites for row in occupations)
        or len(times) != points
        or any(len(rows) != points for rows in (energy, doublon, occupations, states))
        or not states
        or len(states[0]) != expected_state_width
        or any(len(row) != expected_state_width for row in states)
        or not all(math.isfinite(value) for value in times + energy + doublon)
        or not all(math.isfinite(value) for row in occupations for value in row)
        or not all(
            math.isfinite(value.real) and math.isfinite(value.imag)
            for row in states
            for value in row
        )
    ):
        raise ProspectiveExecutionError(
            "exact-reference computation shapes, Hilbert-space dimension, or values are invalid"
        )
    if (
        _canonical_sha(times) != order.reference.grid["ordered_times_sha256"]
        or any(
            not math.isclose(
                math.fsum(abs(value) ** 2 for value in row),
                1.0,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            )
            for row in states
        )
    ):
        raise ProspectiveExecutionError(
            "exact-reference computation grid/state normalization diverged"
        )
    return times, energy, doublon, occupations, states


def _response_scales(
    *,
    energy: Sequence[float],
    doublon: Sequence[float],
    occupations: Sequence[Sequence[float]],
    prefix_points: int,
) -> dict[str, dict[str, float]]:
    def rms(values: Sequence[float], count: int) -> float:
        baseline = float(values[0])
        return math.sqrt(
            math.fsum((float(value) - baseline) ** 2 for value in values[:count])
            / float(count)
        )

    def site_rms(count: int) -> float:
        baseline = tuple(float(value) for value in occupations[0])
        excursions = (
            max(
                abs(float(row[index]) - baseline[index])
                for index in range(len(baseline))
            )
            for row in occupations[:count]
        )
        return math.sqrt(
            math.fsum(value * value for value in excursions) / float(count)
        )

    result = {
        horizon: {
            "energy_rms_excursion": rms(energy, count),
            "doublon_rms_excursion": rms(doublon, count),
            "max_site_occupation_rms_excursion": site_rms(count),
        }
        for horizon, count in (
            ("t5", int(prefix_points)),
            ("t20", len(energy)),
        )
    }
    if not all(
        math.isfinite(value) and value > 0.0
        for row in result.values()
        for value in row.values()
    ):
        raise ProspectiveExecutionError(
            "exact-reference response scales must be finite and positive"
        )
    return result


def materialize_exact_reference(
    order: ExactReferenceWorkOrder,
    *,
    repository_root: str | Path,
    compute: Callable[[ExactReferenceWorkOrder], ExactReferenceComputation] = (
        compute_exact_reference
    ),
) -> ExactReferenceReceipt:
    """Execute and atomically seal one content-addressed exact reference."""

    root = Path(repository_root).resolve()
    reference, seed, epoch = _canonical_reference_and_seed(order.request)
    if (
        order.reference != reference
        or order.seed != seed
        or dict(order.propagation) != propagation_policy(reference)
    ):
        raise ProspectiveExecutionError(
            "exact-reference work order changed before execution"
        )
    source_path = _repository_file(
        root,
        order.source_epoch_receipt_path,
        order.source_epoch_receipt_sha256,
        label="source epoch receipt",
    )
    source_receipt = SourceEpochReceipt.load(
        source_path, repository_root=root, expected_epoch=epoch
    )
    seed_path = _repository_file(
        root,
        order.seed_receipt_path,
        order.seed_receipt_sha256,
        label="seed qualification receipt",
    )
    seed_receipt = SeedQualificationReceipt.load(
        seed_path,
        repository_root=root,
        expected_seed=seed,
        expected_source_epoch=epoch,
    )
    rebuilt = build_work_order(
        order.request,
        source_epoch_receipt=source_receipt,
        seed_receipt=seed_receipt,
    )
    if rebuilt != order:
        raise ProspectiveExecutionError(
            "exact-reference work order digest changed before execution"
        )

    # Default computation resolves the repository-relative seed from cwd.  A
    # package launcher always enters repository root; preserve the same rule
    # for direct execution without leaking a process-wide cwd change to callers.
    if compute is compute_exact_reference:
        seed_absolute = root / order.seed_artifact_path

        def bound_compute(current: ExactReferenceWorkOrder) -> ExactReferenceComputation:
            rebound = ExactReferenceWorkOrder(
                **{
                    **current.__dict__,
                    "seed_artifact_path": seed_absolute.as_posix(),
                }
            )
            return compute_exact_reference(rebound)

        result = bound_compute(order)
    else:
        result = compute(order)
    times, energy, doublon, occupations, states = _normalize_computation(
        result, order=order
    )
    materialized_key = reference.materialized_key(seed_receipt.artifact.sha256)
    artifact_path = (root / str(order.request.inputs["artifact_path"])).resolve()
    try:
        artifact_path.relative_to(root)
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "exact-reference artifact path escapes repository"
        ) from exc
    if artifact_path.name != "manifest.json" or artifact_path.parent.name != materialized_key:
        raise ProspectiveExecutionError(
            "exact-reference artifact path is not its content address"
        )
    data_path = artifact_path.with_name("reference_data.json")
    receipt_path = artifact_path.with_name("exact_reference_receipt.json")
    data = {
        "schema": EXACT_REFERENCE_DATA_V1,
        "reference_identity_sha256": reference.identity_sha256,
        "materialized_reference_key": materialized_key,
        "observable_schema": reference.observable_schema,
        "ordered_times_sha256": reference.grid["ordered_times_sha256"],
        "time": list(times),
        "energy": list(energy),
        "doublon": list(doublon),
        "site_occupations": [list(row) for row in occupations],
        "statevector_real": [
            [float(value.real) for value in row] for row in states
        ],
        "statevector_imag": [
            [float(value.imag) for value in row] for row in states
        ],
    }
    _atomic_write_immutable(data_path, _pretty_bytes(data))
    scales = _response_scales(
        energy=energy,
        doublon=doublon,
        occupations=occupations,
        prefix_points=int(reference.grid["prefix_points"]),
    )
    scales_sha = _canonical_sha(scales)
    contract = _reference_contract(reference, seed)
    manifest = _seal(
        {
            "schema": EXACT_REFERENCE_MANIFEST_V1,
            "worker_schema": EXACT_REFERENCE_WORKER_SCHEMA_V1,
            "action_id": order.request.action_id,
            "execution_request_sha256": order.request.digest,
            "work_order_sha256": order.work_order_sha256,
            "reference_contract": contract,
            "materialized_reference_key": materialized_key,
            "source_epoch_identity_sha256": epoch.identity_sha256,
            "seed_artifact_sha256": seed_receipt.artifact.sha256,
            "data_sha256": _file_sha(data_path),
            "response_scales": scales,
            "response_scales_sha256": scales_sha,
            "propagation_policy": dict(order.propagation),
            "qiskit_compilation_used": False,
            "reporting_only": True,
        },
        "exact_manifest_sha256",
    )
    _atomic_write_immutable(artifact_path, _pretty_bytes(manifest))
    data_relative = data_path.relative_to(root).as_posix()
    manifest_relative = artifact_path.relative_to(root).as_posix()
    receipt = _seal(
        {
            "schema": EXACT_REFERENCE_RECEIPT_V1,
            "reference_id": reference.reference_id,
            "reference_identity_sha256": reference.identity_sha256,
            "lane_id": reference.lane_id,
            "cell_id": reference.cell_id,
            "seed_id": seed.seed_id,
            "seed_artifact_sha256": seed_receipt.artifact.sha256,
            "manifest": {
                "path": manifest_relative,
                "sha256": _file_sha(artifact_path),
            },
            "data": {"path": data_relative, "sha256": _file_sha(data_path)},
            "source_epoch_identity_sha256": epoch.identity_sha256,
            "materialized_reference_key": materialized_key,
            "reference_contract": contract,
            "response_scales": scales,
            "response_scales_sha256": scales_sha,
        },
        "exact_reference_receipt_sha256",
    )
    _atomic_write_immutable(receipt_path, _pretty_bytes(receipt))
    return ExactReferenceReceipt.load(
        receipt_path,
        repository_root=root,
        expected_reference=reference,
        expected_seed_identity=seed,
        expected_seed_receipt=seed_receipt,
        expected_source_epoch=epoch,
    )


def materialization_request(request: ExecutionRequest) -> dict[str, Any]:
    """Authenticate one bootstrap exact action and return an inert recipe."""

    if not isinstance(request, ExecutionRequest):
        raise ProspectiveExecutionError(
            "exact-reference materialization requires a typed execution request"
        )
    reference, seed, epoch = _canonical_reference_and_seed(request)
    return {
        "schema": "paper_ii_exact_reference_operational_materialization_request_v1",
        "status": "operational_materialization_ready",
        "execution_request": request.to_json_dict(),
        "execution_request_sha256": request.digest,
        "reference_identity_sha256": reference.identity_sha256,
        "seed_identity_sha256": seed.identity_sha256,
        "source_epoch_identity_sha256": epoch.identity_sha256,
        "work_order_builder": (
            "pipelines.time_dynamics.prospective_exact_reference_worker:"
            "build_work_order"
        ),
        "package_builder": (
            "pipelines.time_dynamics.prospective_exact_reference_chtc_package:"
            "materialize_exact_reference_action_package"
        ),
        "runner_module": (
            "pipelines.time_dynamics.runners.build_prospective_exact_reference"
        ),
        "required_receipts": [
            "source_epoch_receipt",
            "seed_qualification_receipt",
        ],
        "scheduler_contacted": False,
        "scheduler_submission_authorized": False,
        "production_submission_enabled": False,
        "qiskit_compilation_requested": False,
    }


__all__ = [
    "EXACT_REFERENCE_WORKER_SCHEMA_V1",
    "ExactReferenceComputation",
    "ExactReferenceWorkOrder",
    "PROPAGATION_POLICY_SCHEMA_V1",
    "WORK_ORDER_SCHEMA_V1",
    "build_work_order",
    "compute_exact_reference",
    "load_work_order",
    "materialization_request",
    "materialize_exact_reference",
    "propagation_policy",
    "write_work_order",
]
