"""Source-locked seed worker for the prospective Paper-II Hubbard-L4 lane.

The worker accepts only the one canonical Hubbard-L4 seed action emitted by
the frozen prospective planner.  Its fixed structure is supplied by
``pure_hubbard_fixed_vqe``: two complete, repository-ordered repetitions of
the fixed-spin-sector ``uccsd_qeb_hva_blocks`` pool, executed as grouped exact
logical generators.  Exact data is used only for the offline seed gate and
diagnostic infidelity, never by a later dynamics controller.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from pipelines.time_dynamics.fixed_vqe_conditioning import _normalize_state, _statevector_payload
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    LANE_HUBBARD,
    SeedIdentity,
    SourceEpochIdentity,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    EXEC_HUBBARD_SEED,
    SEED_ARTIFACT_V1,
    SEED_ENERGY_GATE,
    SEED_QUALIFICATION_RECEIPT_V1,
    ExecutionRequest,
    ProspectiveExecutionError,
    SeedQualificationReceipt,
    SourceEpochReceipt,
    executor_bindings,
    resolve_bootstrap_action,
)
from pipelines.time_dynamics.pure_hubbard_fixed_vqe import (
    PURE_HUBBARD_GENERATOR_EXECUTION,
    PURE_HUBBARD_GENERATOR_FILTER,
    PURE_HUBBARD_GENERATOR_ORDER,
    PURE_HUBBARD_GENERATOR_POOL,
    PureHubbardFixedVQEConfig,
    PureHubbardFixedVQEProblem,
    build_pure_hubbard_fixed_vqe_problem,
    qualify_pure_hubbard_fixed_vqe_result,
)
from src.quantum.ansatz_parameterization import expand_legacy_logical_theta, serialize_layout
from src.quantum.vqe_latex_python_pairs import VQEResult, hamiltonian_matrix, vqe_minimize


WORK_ORDER_SCHEMA_V1 = "paper_ii_hubbard_l4_seed_work_order_v1"
ARCHITECTURE_POLICY_SCHEMA_V1 = "paper_ii_hubbard_l4_seed_architecture_policy_v1"
ARCHITECTURE_POLICY_ID = "two_complete_ordered_spin_sector_uccsd_qeb_hva_layers_v1"
SEED_WORKER_SCHEMA_V1 = "paper_ii_hubbard_l4_seed_worker_v1"
SEED_SUBJECT_KIND = "paper_ii_generator_derived_fixed_structure_vqe_v1"
EXPECTED_UNIQUE_GENERATOR_COUNT = 113
EXPECTED_LOGICAL_PARAMETER_COUNT = 226
EXPECTED_SECTOR_DIMENSION = 36


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(payload)
    if key in result:
        raise ValueError(f"payload already contains {key}")
    result[key] = _canonical_sha(result)
    return result


def _verify_seal(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    digest = str(payload.get(key) or "")
    core = {name: value for name, value in payload.items() if name != key}
    if digest != _canonical_sha(core):
        raise ProspectiveExecutionError(f"{label} self digest mismatch")
    return digest


def _pretty_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ProspectiveExecutionError(
                f"refusing to overwrite immutable seed evidence: {path}"
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


def architecture_policy() -> dict[str, Any]:
    """Return the digest-bound pure-Hubbard fixed-structure policy."""

    payload = {
        "schema": ARCHITECTURE_POLICY_SCHEMA_V1,
        "policy_id": ARCHITECTURE_POLICY_ID,
        "generator_pool": PURE_HUBBARD_GENERATOR_POOL,
        "generator_filter": PURE_HUBBARD_GENERATOR_FILTER,
        "generator_order": PURE_HUBBARD_GENERATOR_ORDER,
        "generator_execution": PURE_HUBBARD_GENERATOR_EXECUTION,
        "layer_repeats": 2,
        "layer_relation": "two_identical_complete_ordered_filtered_pool_layers",
        "parameterization_mode": "logical_shared",
        "expected_unique_generator_count": EXPECTED_UNIQUE_GENERATOR_COUNT,
        "expected_logical_parameter_count": EXPECTED_LOGICAL_PARAMETER_COUNT,
        "expected_sector_dimension": EXPECTED_SECTOR_DIMENSION,
        "optimizer": {
            "method": "SLSQP",
            "maxiter": 1800,
            "restarts": 4,
            "zero_first_restart": True,
            "initial_point_stddev": 0.3,
            "bounds": [-math.pi, math.pi],
            "energy_backend": "legacy",
        },
        "selection": "fixed_before_inner_vqe_no_architecture_search",
        "conditioning_search_used": False,
        "snapshot_conditioning_used": False,
        "fs_radius_tuning_used": False,
        "qiskit_compilation_used": False,
    }
    payload["architecture_policy_sha256"] = _canonical_sha(payload)
    return payload


def _source_epoch_from_request(request: ExecutionRequest) -> SourceEpochIdentity:
    raw = request.inputs.get("source_epoch")
    if not isinstance(raw, Mapping):
        raise ProspectiveExecutionError("Hubbard seed request has no source epoch")
    try:
        epoch = SourceEpochIdentity(
            lane_id=str(raw.get("lane_id") or ""),
            source_epoch_id=str(raw.get("source_epoch_id") or ""),
            source_revision=raw.get("source_revision"),
            source_archive_sha256=raw.get("source_archive_sha256"),
        )
    except ValueError as exc:
        raise ProspectiveExecutionError(
            "Hubbard seed request source epoch is invalid"
        ) from exc
    if raw.get("source_epoch_identity_sha256") != epoch.identity_sha256:
        raise ProspectiveExecutionError("Hubbard seed request source epoch digest mismatch")
    if epoch.lane_id != LANE_HUBBARD:
        raise ProspectiveExecutionError("Hubbard seed request crosses lanes")
    return epoch


def validate_canonical_seed_request(request: ExecutionRequest) -> SeedIdentity:
    """Authenticate the sole Hubbard-L4 request against the frozen planner."""

    if (
        request.lane_id != LANE_HUBBARD
        or request.kind != "build_seed"
        or request.binding_id != EXEC_HUBBARD_SEED
        or request.exact_reference_blind
        or request.attempt != 0
        or request.planner_sequence != 0
        or request.requested_memory_gib is not None
        or request.arm is not None
    ):
        raise ProspectiveExecutionError("request is not the canonical Hubbard seed action")
    epoch = _source_epoch_from_request(request)
    registry = build_identity_registry()
    lane_seeds = tuple(seed for seed in registry.seeds if seed.lane_id == LANE_HUBBARD)
    if len(lane_seeds) != 1:
        raise ProspectiveExecutionError("frozen matrix does not contain exactly one Hubbard seed")
    seed = lane_seeds[0]
    if seed.seed_id != request.inputs.get("seed_id"):
        raise ProspectiveExecutionError("request does not name the canonical Hubbard seed")
    plan = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(source_epochs={LANE_HUBBARD: epoch}),
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    actions = tuple(row for row in plan.ready_actions if row.action_id == request.action_id)
    if len(actions) != 1:
        raise ProspectiveExecutionError("request is not emitted by the canonical planner")
    canonical = resolve_bootstrap_action(actions[0])
    if canonical.to_json_dict() != request.to_json_dict():
        raise ProspectiveExecutionError("Hubbard seed request diverges from canonical inputs")
    if (
        seed.model != "hubbard"
        or seed.initializer != "generator_derived_fixed_structure_vqe_v1"
        or seed.restarts != 4
        or seed.energy_gate != SEED_ENERGY_GATE
        or seed.optimizer_seed is None
    ):
        raise ProspectiveExecutionError("canonical Hubbard seed identity changed")
    return seed


@dataclass(frozen=True)
class HubbardSeedWorkOrder:
    request: ExecutionRequest
    seed: SeedIdentity
    source_epoch_receipt_path: str
    source_epoch_receipt_sha256: str
    architecture: Mapping[str, Any]
    work_order_sha256: str

    def to_json_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema": WORK_ORDER_SCHEMA_V1,
            "execution_request": self.request.to_json_dict(),
            "seed_identity": self.seed.to_json_dict(),
            "source_epoch_receipt": {
                "path": self.source_epoch_receipt_path,
                "sha256": self.source_epoch_receipt_sha256,
            },
            "architecture_policy": dict(self.architecture),
            "production_submission_enabled": False,
        }
        if include_digest:
            payload["work_order_sha256"] = self.work_order_sha256
        return payload


def build_work_order(
    request: ExecutionRequest,
    *,
    source_epoch_receipt: SourceEpochReceipt,
) -> HubbardSeedWorkOrder:
    seed = validate_canonical_seed_request(request)
    epoch = _source_epoch_from_request(request)
    if (
        source_epoch_receipt.source_epoch != epoch
        or source_epoch_receipt.lane_id != LANE_HUBBARD
    ):
        raise ProspectiveExecutionError("source epoch receipt/request mismatch")
    policy = architecture_policy()
    core = {
        "schema": WORK_ORDER_SCHEMA_V1,
        "execution_request": request.to_json_dict(),
        "seed_identity": seed.to_json_dict(),
        "source_epoch_receipt": {
            "path": source_epoch_receipt.receipt_path,
            "sha256": source_epoch_receipt.receipt_sha256,
        },
        "architecture_policy": policy,
        "production_submission_enabled": False,
    }
    return HubbardSeedWorkOrder(
        request=request,
        seed=seed,
        source_epoch_receipt_path=source_epoch_receipt.receipt_path,
        source_epoch_receipt_sha256=source_epoch_receipt.receipt_sha256,
        architecture=policy,
        work_order_sha256=_canonical_sha(core),
    )


def load_work_order(
    path: str | Path, *, repository_root: str | Path
) -> tuple[HubbardSeedWorkOrder, SourceEpochReceipt]:
    root = Path(repository_root).resolve()
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    payload = json.loads(source.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != WORK_ORDER_SCHEMA_V1:
        raise ProspectiveExecutionError("unsupported Hubbard seed work order")
    work_sha = _verify_seal(payload, "work_order_sha256", label="Hubbard seed work order")
    request_row = payload.get("execution_request")
    source_row = payload.get("source_epoch_receipt")
    if not isinstance(request_row, Mapping) or not isinstance(source_row, Mapping):
        raise ProspectiveExecutionError("Hubbard seed work order is incomplete")
    request = ExecutionRequest.from_mapping(request_row)
    seed = validate_canonical_seed_request(request)
    if payload.get("seed_identity") != seed.to_json_dict():
        raise ProspectiveExecutionError("Hubbard seed work order identity diverged")
    if payload.get("architecture_policy") != architecture_policy():
        raise ProspectiveExecutionError("Hubbard seed architecture policy diverged")
    receipt_path = str(source_row.get("path") or "")
    receipt_sha = str(source_row.get("sha256") or "")
    receipt_source = (root / receipt_path).resolve(strict=True)
    if _file_sha(receipt_source) != receipt_sha:
        raise ProspectiveExecutionError("source epoch receipt file digest mismatch")
    epoch = _source_epoch_from_request(request)
    receipt = SourceEpochReceipt.load(
        receipt_source, repository_root=root, expected_epoch=epoch
    )
    result = build_work_order(request, source_epoch_receipt=receipt)
    if result.work_order_sha256 != work_sha:
        raise ProspectiveExecutionError("Hubbard seed work order canonical digest changed")
    return result, receipt


def write_work_order(path: str | Path, order: HubbardSeedWorkOrder) -> Path:
    target = Path(path)
    _atomic_write_immutable(target, _pretty_bytes(order.to_json_dict()))
    return target


@dataclass(frozen=True)
class SeedComputation:
    variational_energy: float
    exact_energy: float
    exact_state_infidelity: float
    runtime_payload: Mapping[str, Any]
    optimizer_receipt: Mapping[str, Any]
    architecture_binding: Mapping[str, Any]


def _config_from_seed(seed: SeedIdentity) -> PureHubbardFixedVQEConfig:
    physical = dict(seed.physical)
    hopping = float(physical["t_hop"])
    return PureHubbardFixedVQEConfig(
        num_sites=int(physical["lattice_sites"]),
        t=hopping,
        u=float(physical["U_over_t"]) * hopping,
        n_up=int(physical["n_up"]),
        n_down=int(physical["n_down"]),
        ordering=str(physical["ordering"]),
        boundary=str(physical["boundary"]),
        reps=2,
        restarts=int(seed.restarts or 0),
        optimizer_seed=int(seed.optimizer_seed or -1),
        optimizer_method="SLSQP",
        maxiter=1800,
        initial_point_stddev=0.3,
        energy_gate=float(seed.energy_gate or math.nan),
    )


def _sector_basis_indices(config: PureHubbardFixedVQEConfig) -> tuple[int, ...]:
    if str(config.ordering) == "blocked":
        alpha = tuple(range(int(config.num_sites)))
        beta = tuple(range(int(config.num_sites), 2 * int(config.num_sites)))
    else:
        alpha = tuple(2 * site for site in range(int(config.num_sites)))
        beta = tuple(2 * site + 1 for site in range(int(config.num_sites)))
    return tuple(
        index
        for index in range(1 << int(config.total_qubits))
        if sum((index >> qubit) & 1 for qubit in alpha) == int(config.n_up)
        and sum((index >> qubit) & 1 for qubit in beta) == int(config.n_down)
    )


def _exact_sector_ground_state(
    problem: PureHubbardFixedVQEProblem,
) -> tuple[float, np.ndarray]:
    matrix = np.asarray(hamiltonian_matrix(problem.hamiltonian), dtype=complex)
    basis = _sector_basis_indices(problem.config)
    if len(basis) != int(problem.sector_dimension):
        raise ProspectiveExecutionError("pure-Hubbard sector dimension changed")
    subspace = matrix[np.ix_(basis, basis)]
    eigenvalues, eigenvectors = np.linalg.eigh(subspace)
    minimum = int(np.argmin(np.real(eigenvalues)))
    state = np.zeros(matrix.shape[0], dtype=complex)
    state[np.asarray(basis, dtype=int)] = eigenvectors[:, minimum]
    return float(np.real(eigenvalues[minimum])), _normalize_state(
        state, name="prospective_hubbard_seed_exact_state"
    )


def _assert_problem_contract(
    problem: PureHubbardFixedVQEProblem,
    config: PureHubbardFixedVQEConfig,
) -> None:
    if (
        problem.config != config
        or int(problem.unique_generator_count) != EXPECTED_UNIQUE_GENERATOR_COUNT
        or int(problem.ansatz.num_parameters) != EXPECTED_LOGICAL_PARAMETER_COUNT
        or int(problem.sector_dimension) != EXPECTED_SECTOR_DIMENSION
        or len(problem.generator_terms) != EXPECTED_LOGICAL_PARAMETER_COUNT
        or len(str(problem.generator_pool_sha256)) != 64
        or any(
            str(term.execution_mode) != "grouped_exact"
            for term in problem.generator_terms
        )
    ):
        raise ProspectiveExecutionError("pure-Hubbard fixed architecture contract changed")


def compute_seed(order: HubbardSeedWorkOrder) -> SeedComputation:
    """Execute the four-restart fixed VQE; intended for CHTC, not a laptop."""

    seed = order.seed
    policy = architecture_policy()
    if dict(order.architecture) != policy:
        raise ProspectiveExecutionError("Hubbard seed architecture policy changed")
    config = _config_from_seed(seed)
    if config.to_json_dict()["initializer"] != seed.initializer:
        raise ProspectiveExecutionError("Hubbard seed initializer changed")
    problem = build_pure_hubbard_fixed_vqe_problem(config)
    _assert_problem_contract(problem, config)
    optimizer = dict(policy["optimizer"])
    result = vqe_minimize(
        problem.hamiltonian,
        problem.ansatz,
        problem.reference_state,
        restarts=int(seed.restarts or 0),
        seed=int(seed.optimizer_seed or -1),
        initial_point=np.zeros(problem.ansatz.num_parameters, dtype=float),
        use_initial_point_first_restart=True,
        initial_point_stddev=float(optimizer["initial_point_stddev"]),
        method=str(optimizer["method"]),
        maxiter=int(optimizer["maxiter"]),
        bounds=tuple(float(value) for value in optimizer["bounds"]),
        energy_backend=str(optimizer["energy_backend"]),
    )
    summaries = tuple(dict(row) for row in (result.restart_summaries or ()))
    if len(summaries) != 4 or [row.get("restart_index") for row in summaries] != [1, 2, 3, 4]:
        raise ProspectiveExecutionError("seed VQE did not complete four deterministic restarts")
    theta = np.asarray(result.theta, dtype=float).reshape(-1)
    if theta.size != problem.ansatz.num_parameters or not np.all(np.isfinite(theta)):
        raise ProspectiveExecutionError("seed VQE returned invalid parameters")
    qualification = qualify_pure_hubbard_fixed_vqe_result(problem, result)
    if not qualification.qualified:
        raise ProspectiveExecutionError(
            "seed fails the hard Delta E <= 1e-6 gate: "
            f"{qualification.absolute_energy_error:.16g}"
        )
    exact_energy, exact_state = _exact_sector_ground_state(problem)
    if not math.isclose(exact_energy, problem.exact_energy, rel_tol=0.0, abs_tol=1.0e-12):
        raise ProspectiveExecutionError("pure-Hubbard exact energy implementations disagree")
    prepared = _normalize_state(
        problem.ansatz.prepare_state(theta, problem.reference_state),
        name="prospective_hubbard_seed_state",
    )
    fidelity = float(abs(np.vdot(exact_state, prepared)) ** 2)
    infidelity = float(min(1.0, max(0.0, 1.0 - fidelity)))
    if not math.isfinite(infidelity):
        raise ProspectiveExecutionError("seed VQE returned invalid exact-state infidelity")

    layout_payload = serialize_layout(problem.ansatz.layout)
    theta_runtime = expand_legacy_logical_theta(theta, problem.ansatz.layout)
    settings = config.to_json_dict()
    settings.update(
        {
            "adapt_pool": PURE_HUBBARD_GENERATOR_POOL,
            "dv": 0.0,
            "sector_n_dn": int(config.n_down),
            "include_zero_point": False,
        }
    )
    runtime_labels = [str(term.label) for term in problem.generator_terms]
    runtime_payload = {
        "schema_version": SEED_WORKER_SCHEMA_V1,
        "pipeline": "prospective_hubbard_seed_worker",
        "settings": settings,
        "ansatz_input_state": _statevector_payload(
            problem.reference_state,
            source="pure_hubbard_hartree_fock_reference_state",
            handoff_state_kind="reference_state",
        ),
        "initial_state": _statevector_payload(
            prepared,
            source="paper_ii_generator_derived_fixed_structure_vqe",
            handoff_state_kind="prepared_state",
        ),
        "adapt_vqe": {
            "pool_type": PURE_HUBBARD_GENERATOR_POOL,
            "structure_locked": True,
            "fixed_scaffold_kind": SEED_SUBJECT_KIND,
            "parameterization": layout_payload,
            "parameterization_mode": "logical_shared",
            "parameterization_execution_mode": "logical_shared",
            "num_parameters": int(problem.ansatz.num_parameters),
            "logical_num_parameters": int(problem.ansatz.num_parameters),
            "runtime_parameter_count": int(problem.ansatz.layout.runtime_parameter_count),
            "ansatz_depth": int(len(problem.generator_terms)),
            "operators": runtime_labels,
            "optimal_point": [float(value) for value in theta_runtime],
            "logical_optimal_point": [float(value) for value in theta],
            "energy": float(result.energy),
            "num_particles": {"n_up": int(config.n_up), "n_dn": int(config.n_down)},
            "fixed_scaffold_metadata": {
                "schema_version": 1,
                "route_family": "paper_ii_pure_hubbard_fixed_structure_v1",
                "subject_kind": SEED_SUBJECT_KIND,
                "structure_locked": True,
                "operator_count": int(len(problem.generator_terms)),
                "runtime_term_count": int(problem.ansatz.layout.runtime_parameter_count),
                "term_order_id": ARCHITECTURE_POLICY_ID,
                "term_order_basis": PURE_HUBBARD_GENERATOR_ORDER,
                "generator_pool_sha256": problem.generator_pool_sha256,
                "generator_filter": PURE_HUBBARD_GENERATOR_FILTER,
                "generator_execution": PURE_HUBBARD_GENERATOR_EXECUTION,
            },
        },
        "exact_energy": float(problem.exact_energy),
    }
    return SeedComputation(
        variational_energy=float(result.energy),
        exact_energy=float(problem.exact_energy),
        exact_state_infidelity=infidelity,
        runtime_payload=runtime_payload,
        optimizer_receipt={
            "method": str(optimizer["method"]),
            "maxiter": int(optimizer["maxiter"]),
            "restarts": 4,
            "optimizer_seed": int(seed.optimizer_seed),
            "zero_first_restart": True,
            "best_restart": int(result.best_restart) + 1,
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "restart_summaries": list(summaries),
        },
        architecture_binding={
            "policy": policy,
            "generator_pool_sha256": problem.generator_pool_sha256,
            "unique_generator_count": int(problem.unique_generator_count),
            "logical_parameter_count": int(problem.ansatz.num_parameters),
            "runtime_parameter_count": int(problem.ansatz.layout.runtime_parameter_count),
            "sector_dimension": int(problem.sector_dimension),
            "parameterization_sha256": _canonical_sha(layout_payload),
            "theta_logical_sha256": hashlib.sha256(
                np.asarray(theta, dtype="<f8").tobytes()
            ).hexdigest(),
            "theta_runtime_sha256": hashlib.sha256(
                np.asarray(theta_runtime, dtype="<f8").tobytes()
            ).hexdigest(),
            "exact_reference": {
                "source": "sector_filtered_dense_eigendecomposition",
                "sector_dimension": int(problem.sector_dimension),
            },
            "conditioning_search_used": False,
            "snapshot_conditioning_used": False,
            "qiskit_compilation_used": False,
        },
    )


def materialize_seed(
    order: HubbardSeedWorkOrder,
    *,
    repository_root: str | Path,
    compute: Callable[[HubbardSeedWorkOrder], SeedComputation] = compute_seed,
) -> SeedQualificationReceipt:
    """Run and atomically seal the canonical seed artifact and receipt."""

    root = Path(repository_root).resolve()
    seed = validate_canonical_seed_request(order.request)
    if seed != order.seed or dict(order.architecture) != architecture_policy():
        raise ProspectiveExecutionError("Hubbard seed work order changed before execution")
    epoch = _source_epoch_from_request(order.request)
    source_path = (root / order.source_epoch_receipt_path).resolve(strict=True)
    if _file_sha(source_path) != order.source_epoch_receipt_sha256:
        raise ProspectiveExecutionError("source epoch receipt changed before execution")
    SourceEpochReceipt.load(source_path, repository_root=root, expected_epoch=epoch)
    result = compute(order)
    error = abs(float(result.variational_energy) - float(result.exact_energy))
    if not math.isfinite(error) or error > SEED_ENERGY_GATE:
        raise ProspectiveExecutionError("seed fails the hard Delta E <= 1e-6 gate")
    if not (0.0 <= float(result.exact_state_infidelity) <= 1.0):
        raise ProspectiveExecutionError("seed exact-state infidelity is invalid")
    if result.architecture_binding.get("policy") != architecture_policy():
        raise ProspectiveExecutionError("computed seed architecture policy diverged")
    optimizer_execution_id = f"{order.request.execution_id}::vqe4"
    artifact_core = {
        **dict(result.runtime_payload),
        "schema": SEED_ARTIFACT_V1,
        "seed_id": seed.seed_id,
        "seed_identity_sha256": seed.identity_sha256,
        "lane_id": seed.lane_id,
        "model": seed.model,
        "physical": dict(seed.physical),
        "seed_policy": seed.policy,
        "initializer": seed.initializer,
        "restarts": seed.restarts,
        "optimizer_seed": seed.optimizer_seed,
        "optimizer_execution_id": optimizer_execution_id,
        "source_epoch_identity_sha256": epoch.identity_sha256,
        "variational_energy": float(result.variational_energy),
        "exact_energy": float(result.exact_energy),
        "exact_state_infidelity": float(result.exact_state_infidelity),
        "architecture_binding": dict(result.architecture_binding),
        "optimizer_receipt": dict(result.optimizer_receipt),
        "work_order_sha256": order.work_order_sha256,
        "conditioning_search_used": False,
        "snapshot_conditioning_used": False,
        "qiskit_compilation_used": False,
    }
    artifact = _seal(artifact_core, "seed_artifact_sha256")
    artifact_path = (root / seed.artifact_path).resolve()
    try:
        artifact_relative = artifact_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProspectiveExecutionError("seed artifact escapes repository root") from exc
    _atomic_write_immutable(artifact_path, _pretty_bytes(artifact))
    receipt_path = artifact_path.with_name("seed_qualification_receipt.json")
    receipt_core = {
        "schema": SEED_QUALIFICATION_RECEIPT_V1,
        "seed_id": seed.seed_id,
        "seed_identity_sha256": seed.identity_sha256,
        "artifact": {"path": artifact_relative, "sha256": _file_sha(artifact_path)},
        "energy_error": error,
        "exact_state_infidelity": float(result.exact_state_infidelity),
        "initializer": seed.initializer,
        "restarts": seed.restarts,
        "optimizer_seed": seed.optimizer_seed,
        "optimizer_execution_id": optimizer_execution_id,
        "variational_energy": float(result.variational_energy),
        "exact_energy": float(result.exact_energy),
        "source_epoch_identity_sha256": epoch.identity_sha256,
        "work_order_sha256": order.work_order_sha256,
    }
    receipt = _seal(receipt_core, "seed_receipt_sha256")
    _atomic_write_immutable(receipt_path, _pretty_bytes(receipt))
    return SeedQualificationReceipt.load(
        receipt_path,
        repository_root=root,
        expected_seed=seed,
        expected_source_epoch=epoch,
    )


__all__ = [
    "ARCHITECTURE_POLICY_ID",
    "ARCHITECTURE_POLICY_SCHEMA_V1",
    "HubbardSeedWorkOrder",
    "SEED_SUBJECT_KIND",
    "SEED_WORKER_SCHEMA_V1",
    "SeedComputation",
    "WORK_ORDER_SCHEMA_V1",
    "architecture_policy",
    "build_work_order",
    "compute_seed",
    "load_work_order",
    "materialize_seed",
    "validate_canonical_seed_request",
    "write_work_order",
]
