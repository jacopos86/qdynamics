"""Source-locked seed worker for the prospective Paper-II HH product lane.

This module materializes only the six canonical Hubbard--Holstein seed
identities.  The fixed architecture is deliberately explicit: three identical
complete ordered Hamiltonian parent layers (``hop_layer``, ``onsite_layer``,
``phonon_layer``, ``eph_layer``).  It does not run the conditioning search,
fit driven snapshots, or use exact data in the VQE optimizer or any later
dynamics controller.
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

from pipelines.scaffold.hh_fixed_manifold_loader import _make_replay_run_cfg
from pipelines.scaffold.hh_vqe_from_adapt_family import _build_hh_hamiltonian
from pipelines.static_adapt.builders.problem_registry import resolve_problem_context
from pipelines.time_dynamics.fixed_vqe_conditioning import (
    CONSTRUCTION_MODE_CONVENTIONAL,
    FIXED_VQE_STRESS_ROUTE_FAMILY,
    FixedVQEModelConfig,
    GeneratorPoolConfig,
    _exact_ground_reference,
    _normalize_state,
    _problem_request_from_model,
    _statevector_payload,
    architecture_from_layers,
    build_architecture_runtime,
    build_fixed_vqe_generator_pool,
    parent_complete_layer,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    LANE_HH,
    SeedIdentity,
    SourceEpochIdentity,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    EXEC_HH_SEED,
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
from src.quantum.ansatz_parameterization import (
    project_runtime_theta_block_mean,
    serialize_layout,
)
from src.quantum.compiled_polynomial import compile_polynomial_action
from src.quantum.vqe_latex_python_pairs import VQEResult, hamiltonian_matrix


WORK_ORDER_SCHEMA_V1 = "paper_ii_hh_seed_work_order_v1"
ARCHITECTURE_POLICY_SCHEMA_V1 = "paper_ii_hh_seed_architecture_policy_v1"
ARCHITECTURE_POLICY_ID = "three_complete_ordered_hh_hamiltonian_layers_v2"
SEED_WORKER_SCHEMA_V1 = "paper_ii_hh_seed_worker_v1"
SEED_SUBJECT_KIND = "paper_ii_generator_derived_fixed_structure_vqe_v1"
HH_SEED_LAYER_PARENTS = (
    "hop_layer",
    "onsite_layer",
    "phonon_layer",
    "eph_layer",
)
HH_SEED_LAYER_REPEATS = 3


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
    """Return the frozen, digest-bound conventional seed architecture."""

    payload = {
        "schema": ARCHITECTURE_POLICY_SCHEMA_V1,
        "policy_id": ARCHITECTURE_POLICY_ID,
        "construction_mode": CONSTRUCTION_MODE_CONVENTIONAL,
        "parent_labels": list(HH_SEED_LAYER_PARENTS),
        "layer_repeats": HH_SEED_LAYER_REPEATS,
        "layer_relation": "three_identical_complete_ordered_hamiltonian_parent_layers",
        "pool_key": "full_meta",
        "include_pauli_children": True,
        "include_polyterm_children": True,
        "polyterm_subset_sizes": [2],
        "parameterization_mode": "per_pauli_term",
        "optimizer": {
            "method": "L-BFGS-B",
            "maxiter": 20000,
            "restarts": 4,
            "zero_first_restart": True,
            "initial_point_stddev": 0.4,
            "bounds": [-math.pi, math.pi],
            "energy_backend": "one_apply_compiled",
            "gradient_backend": "compiled_reverse_mode_exact_v1",
            "max_function_evaluations": 20000,
            "ftol": 1.0e-13,
            "gtol": 1.0e-8,
            "optimizer_exact_reference_used": False,
        },
        "selection": "fixed_before_inner_vqe_no_architecture_search",
        "conditioning_search_used": False,
        "snapshot_conditioning_used": False,
        "fs_radius_tuning_used": False,
    }
    payload["architecture_policy_sha256"] = _canonical_sha(payload)
    return payload


def _minimize_seed_vqe(
    static_poly: Any,
    executor: Any,
    psi_ref: np.ndarray,
    *,
    optimizer: Mapping[str, Any],
    restarts: int,
    optimizer_seed: int,
) -> VQEResult:
    """Run the frozen HH optimizer with an exact compiled adjoint gradient.

    The objective and gradient use only the declared Hamiltonian, the fixed
    ansatz, and its reference state.  The separately constructed exact ground
    state is intentionally absent from this interface and remains restricted
    to the post-optimization qualification gate and diagnostic infidelity.
    """

    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover - the production image pins SciPy
        raise ProspectiveExecutionError(
            "HH seed reverse-mode VQE requires scipy.optimize.minimize"
        ) from exc

    if (
        str(optimizer.get("method")) != "L-BFGS-B"
        or str(optimizer.get("energy_backend")) != "one_apply_compiled"
        or str(optimizer.get("gradient_backend"))
        != "compiled_reverse_mode_exact_v1"
        or optimizer.get("optimizer_exact_reference_used") is not False
        or str(getattr(executor, "parameterization_mode", ""))
        != "per_pauli_term"
    ):
        raise ProspectiveExecutionError("HH seed optimizer policy is unsupported")
    if int(restarts) != 4:
        raise ProspectiveExecutionError("HH seed optimizer requires four restarts")

    parameter_count = int(getattr(executor, "runtime_parameter_count", 0))
    if (
        parameter_count <= 0
        or int(getattr(executor, "num_parameters", 0)) != parameter_count
    ):
        raise ProspectiveExecutionError("HH seed executor parameterization is invalid")
    compiled_h = compile_polynomial_action(
        static_poly,
        tol=1.0e-12,
        pauli_action_cache=getattr(executor, "pauli_action_cache", None),
    )

    bounds_raw = optimizer.get("bounds")
    if not isinstance(bounds_raw, Sequence) or len(bounds_raw) != 2:
        raise ProspectiveExecutionError("HH seed optimizer bounds are invalid")
    lower, upper = (float(value) for value in bounds_raw)
    bounds = [(lower, upper)] * parameter_count
    maxiter = int(optimizer["maxiter"])
    maxfun = int(optimizer["max_function_evaluations"])
    scipy_options = {
        "maxiter": maxiter,
        "maxfun": maxfun,
        "ftol": float(optimizer["ftol"]),
        "gtol": float(optimizer["gtol"]),
    }
    rng = np.random.default_rng(int(optimizer_seed))
    summaries: list[dict[str, Any]] = []
    best: Any | None = None
    best_restart = -1

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        energy, gradient, _prepared = executor.energy_and_runtime_gradient(
            np.asarray(theta, dtype=float),
            np.asarray(psi_ref, dtype=complex),
            compiled_h,
        )
        if not math.isfinite(float(energy)) or not np.all(np.isfinite(gradient)):
            raise ProspectiveExecutionError(
                "HH seed optimizer produced nonfinite objective data"
            )
        return float(energy), np.asarray(gradient, dtype=float)

    for restart_index in range(int(restarts)):
        if restart_index == 0 and bool(optimizer["zero_first_restart"]):
            initial = np.zeros(parameter_count, dtype=float)
        else:
            initial = float(optimizer["initial_point_stddev"]) * rng.normal(
                size=parameter_count
            )
        result = minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=scipy_options,
        )
        energy = float(result.fun)
        theta = np.asarray(result.x, dtype=float).reshape(-1)
        if (
            not math.isfinite(energy)
            or theta.size != parameter_count
            or not np.all(np.isfinite(theta))
        ):
            raise ProspectiveExecutionError("HH seed optimizer result is invalid")
        summary = {
            "restart_index": restart_index + 1,
            "energy": energy,
            "best_energy": energy,
            "nfev": int(getattr(result, "nfev", 0)),
            "nit": int(getattr(result, "nit", 0)),
            "success": bool(result.success),
            "message": str(result.message),
            "gradient_backend": "compiled_reverse_mode_exact_v1",
            "optimizer_exact_reference_used": False,
        }
        summaries.append(summary)
        if best is None or energy < float(best.fun):
            best = result
            best_restart = restart_index

    if best is None or best_restart < 0:  # pragma: no cover - four restarts above
        raise ProspectiveExecutionError("HH seed optimizer produced no result")
    return VQEResult(
        energy=float(best.fun),
        theta=np.asarray(best.x, dtype=float),
        success=bool(best.success),
        message=str(best.message),
        nfev=int(getattr(best, "nfev", 0)),
        nit=int(getattr(best, "nit", 0)),
        best_restart=int(best_restart),
        restart_summaries=summaries,
    )


def _source_epoch_from_request(request: ExecutionRequest) -> SourceEpochIdentity:
    raw = request.inputs.get("source_epoch")
    if not isinstance(raw, Mapping):
        raise ProspectiveExecutionError("HH seed request has no source epoch")
    try:
        epoch = SourceEpochIdentity(
            lane_id=str(raw.get("lane_id") or ""),
            source_epoch_id=str(raw.get("source_epoch_id") or ""),
            source_revision=raw.get("source_revision"),
            source_archive_sha256=raw.get("source_archive_sha256"),
        )
    except ValueError as exc:
        raise ProspectiveExecutionError("HH seed request source epoch is invalid") from exc
    if raw.get("source_epoch_identity_sha256") != epoch.identity_sha256:
        raise ProspectiveExecutionError("HH seed request source epoch digest mismatch")
    if epoch.lane_id != LANE_HH:
        raise ProspectiveExecutionError("HH seed request crosses lanes")
    return epoch


def validate_canonical_seed_request(request: ExecutionRequest) -> SeedIdentity:
    """Authenticate a request against the current frozen matrix and planner."""

    if (
        request.lane_id != LANE_HH
        or request.kind != "build_seed"
        or request.binding_id != EXEC_HH_SEED
        or request.exact_reference_blind
        or request.attempt != 0
        or request.planner_sequence != 0
        or request.requested_memory_gib is not None
        or request.arm is not None
    ):
        raise ProspectiveExecutionError("request is not a canonical HH seed action")
    epoch = _source_epoch_from_request(request)
    registry = build_identity_registry()
    matching = tuple(
        seed
        for seed in registry.seeds
        if seed.lane_id == LANE_HH and seed.seed_id == request.inputs.get("seed_id")
    )
    if len(matching) != 1:
        raise ProspectiveExecutionError("request does not name one canonical HH seed")
    seed = matching[0]
    plan = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(source_epochs={LANE_HH: epoch}),
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    actions = tuple(row for row in plan.ready_actions if row.action_id == request.action_id)
    if len(actions) != 1:
        raise ProspectiveExecutionError("request is not emitted by the canonical planner")
    canonical = resolve_bootstrap_action(actions[0])
    if canonical.to_json_dict() != request.to_json_dict():
        raise ProspectiveExecutionError("HH seed request diverges from canonical inputs")
    if (
        seed.initializer != "generator_derived_fixed_structure_vqe_v1"
        or seed.restarts != 4
        or seed.energy_gate != SEED_ENERGY_GATE
        or seed.optimizer_seed is None
    ):
        raise ProspectiveExecutionError("canonical HH seed identity changed")
    return seed


@dataclass(frozen=True)
class HHSeedWorkOrder:
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
) -> HHSeedWorkOrder:
    seed = validate_canonical_seed_request(request)
    epoch = _source_epoch_from_request(request)
    if source_epoch_receipt.source_epoch != epoch or source_epoch_receipt.lane_id != LANE_HH:
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
    return HHSeedWorkOrder(
        request=request,
        seed=seed,
        source_epoch_receipt_path=source_epoch_receipt.receipt_path,
        source_epoch_receipt_sha256=source_epoch_receipt.receipt_sha256,
        architecture=policy,
        work_order_sha256=_canonical_sha(core),
    )


def load_work_order(
    path: str | Path, *, repository_root: str | Path
) -> tuple[HHSeedWorkOrder, SourceEpochReceipt]:
    root = Path(repository_root).resolve()
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    payload = json.loads(source.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != WORK_ORDER_SCHEMA_V1:
        raise ProspectiveExecutionError("unsupported HH seed work order")
    work_sha = _verify_seal(payload, "work_order_sha256", label="HH seed work order")
    request_row = payload.get("execution_request")
    source_row = payload.get("source_epoch_receipt")
    if not isinstance(request_row, Mapping) or not isinstance(source_row, Mapping):
        raise ProspectiveExecutionError("HH seed work order is incomplete")
    request = ExecutionRequest.from_mapping(request_row)
    seed = validate_canonical_seed_request(request)
    if payload.get("seed_identity") != seed.to_json_dict():
        raise ProspectiveExecutionError("HH seed work order seed identity diverged")
    policy = architecture_policy()
    if payload.get("architecture_policy") != policy:
        raise ProspectiveExecutionError("HH seed architecture policy diverged")
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
        raise ProspectiveExecutionError("HH seed work order canonical digest changed")
    return result, receipt


def write_work_order(path: str | Path, order: HHSeedWorkOrder) -> Path:
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


def _model_from_seed(seed: SeedIdentity) -> FixedVQEModelConfig:
    physical = dict(seed.physical)
    return FixedVQEModelConfig(
        num_sites=int(physical["lattice_sites"]),
        t=float(physical["t_hop"]),
        u=float(physical["U"]),
        dv=float(physical["staggered_potential"]),
        omega0=float(physical["omega0"]),
        g_ep=float(physical["g"]),
        n_ph_max=int(physical["n_ph_max"]),
        boson_encoding=str(physical["boson_encoding"]),
        ordering=str(physical["ordering"]),
        boundary=str(physical["boundary"]),
        sector_n_up=int(physical["n_up"]),
        sector_n_dn=int(physical["n_down"]),
    )


def compute_seed(order: HHSeedWorkOrder) -> SeedComputation:
    """Execute the fixed architecture VQE; intended for CHTC, not a laptop."""

    seed = order.seed
    policy = architecture_policy()
    if dict(order.architecture) != policy:
        raise ProspectiveExecutionError("HH seed architecture policy changed")
    model = _model_from_seed(seed)
    settings = model.to_settings_payload()
    cfg = _make_replay_run_cfg(
        {"settings": settings, "adapt_vqe": {}},
        artifact_json=Path(seed.artifact_path),
        tag="paper_ii_prospective_hh_seed",
        generator_family="full_meta",
        fallback_family="full_meta",
    )
    static_poly = _build_hh_hamiltonian(cfg)
    resolved = resolve_problem_context(
        _problem_request_from_model(model), hamiltonian=static_poly
    )
    psi_ref = _normalize_state(
        resolved.reference_state.build_state(), name="prospective_hh_seed_reference"
    )
    pool = build_fixed_vqe_generator_pool(
        cfg=cfg,
        h_poly=static_poly,
        model=model,
        pool_config=GeneratorPoolConfig(
            pool_key="full_meta",
            include_pauli_children=True,
            include_polyterm_children=True,
            polyterm_subset_sizes=(2,),
            max_atoms_per_parent=None,
            max_pool_atoms=None,
            priority_parent_labels=HH_SEED_LAYER_PARENTS,
        ),
    )
    base_layer = parent_complete_layer(pool, HH_SEED_LAYER_PARENTS)
    present = {pool.by_id(atom_id).parent_label for atom_id in base_layer}
    if present != set(HH_SEED_LAYER_PARENTS):
        raise ProspectiveExecutionError("canonical HH parent layer is incomplete")
    architecture = architecture_from_layers(
        [list(base_layer) for _ in range(HH_SEED_LAYER_REPEATS)],
        construction_mode=CONSTRUCTION_MODE_CONVENTIONAL,
        construction_notes={
            "origin": "paper_ii_prospective_standard_initializer",
            "parent_labels": list(HH_SEED_LAYER_PARENTS),
            "layer_repeats": HH_SEED_LAYER_REPEATS,
            "architecture_policy_sha256": policy["architecture_policy_sha256"],
        },
    )
    runtime = build_architecture_runtime(architecture, pool=pool)
    optimizer = dict(policy["optimizer"])
    result = _minimize_seed_vqe(
        static_poly,
        runtime.executor,
        psi_ref,
        optimizer=optimizer,
        restarts=int(seed.restarts or 0),
        optimizer_seed=int(seed.optimizer_seed),
    )
    summaries = tuple(dict(row) for row in (result.restart_summaries or ()))
    if len(summaries) != 4 or [row.get("restart_index") for row in summaries] != [1, 2, 3, 4]:
        raise ProspectiveExecutionError("seed VQE did not complete four independent restarts")
    theta = np.asarray(result.theta, dtype=float).reshape(-1)
    if theta.size != runtime.runtime_parameter_count or not np.all(np.isfinite(theta)):
        raise ProspectiveExecutionError("seed VQE returned invalid parameters")
    prepared = _normalize_state(
        runtime.executor.prepare_state(theta, psi_ref), name="prospective_hh_seed_state"
    )
    # Construct the exact reference only after the Hamiltonian-only optimizer
    # has terminated.  This makes the no-leakage boundary structural in
    # addition to keeping exact data out of ``_minimize_seed_vqe``'s interface.
    static_matrix = np.asarray(hamiltonian_matrix(static_poly), dtype=complex)
    exact_energy, exact_state, exact_meta = _exact_ground_reference(
        static_poly=static_poly,
        static_matrix=static_matrix,
        resolved_problem=resolved,
        model=model,
    )
    fidelity = float(abs(np.vdot(exact_state, prepared)) ** 2)
    infidelity = float(min(1.0, max(0.0, 1.0 - fidelity)))
    variational = float(result.energy)
    error = abs(variational - float(exact_energy))
    if not all(math.isfinite(value) for value in (variational, exact_energy, infidelity)):
        raise ProspectiveExecutionError("seed VQE returned nonfinite qualification data")
    if error > SEED_ENERGY_GATE:
        raise ProspectiveExecutionError(
            f"seed fails the hard Delta E <= 1e-6 gate: {error:.16g}"
        )
    layout_payload = serialize_layout(runtime.layout)
    theta_logical = project_runtime_theta_block_mean(theta, runtime.layout)
    runtime_labels = [
        str(term["pauli_exyz"])
        for block in layout_payload["blocks"]
        for term in block["runtime_terms_exyz"]
    ]
    settings["adapt_pool"] = "fixed_scaffold_locked"
    runtime_payload = {
        "schema_version": SEED_WORKER_SCHEMA_V1,
        "pipeline": "prospective_hh_seed_worker",
        "settings": settings,
        "ansatz_input_state": _statevector_payload(
            psi_ref,
            source="hh_canonical_reference_state",
            handoff_state_kind="reference_state",
        ),
        "initial_state": _statevector_payload(
            prepared,
            source="paper_ii_generator_derived_fixed_structure_vqe",
            handoff_state_kind="prepared_state",
        ),
        "adapt_vqe": {
            "pool_type": "fixed_scaffold_locked",
            "structure_locked": True,
            "fixed_scaffold_kind": SEED_SUBJECT_KIND,
            "parameterization": layout_payload,
            "parameterization_mode": "per_pauli_term",
            "parameterization_execution_mode": "per_pauli_term",
            "num_parameters": int(runtime.runtime_parameter_count),
            "logical_num_parameters": int(runtime.layout.logical_parameter_count),
            "ansatz_depth": int(len(runtime.layout.blocks)),
            "optimal_point": [float(value) for value in theta],
            "logical_optimal_point": [float(value) for value in theta_logical],
            "energy": variational,
            "num_particles": {"n_up": model.num_particles[0], "n_dn": model.num_particles[1]},
            "fixed_scaffold_metadata": {
                "schema_version": 1,
                "route_family": FIXED_VQE_STRESS_ROUTE_FAMILY,
                "subject_kind": SEED_SUBJECT_KIND,
                "structure_locked": True,
                "operator_count": int(len(runtime.layout.blocks)),
                "runtime_term_count": int(runtime.runtime_parameter_count),
                "term_order_id": "paper_ii_three_complete_hamiltonian_parent_layers_v2",
                "term_order_basis": "fixed_layered_full_meta_child_occurrences",
                "source_order_runtime_indices": list(range(len(runtime_labels))),
                "source_order_runtime_term_labels_exyz": runtime_labels,
                "runtime_term_labels_exyz": runtime_labels,
                "source_artifact_json": None,
                "source_pool_type": str(pool.pool_key),
            },
        },
        "exact_energy": float(exact_energy),
    }
    return SeedComputation(
        variational_energy=variational,
        exact_energy=float(exact_energy),
        exact_state_infidelity=infidelity,
        runtime_payload=runtime_payload,
        optimizer_receipt={
            "method": str(optimizer["method"]),
            "maxiter": int(optimizer["maxiter"]),
            "max_function_evaluations": int(
                optimizer["max_function_evaluations"]
            ),
            "ftol": float(optimizer["ftol"]),
            "gtol": float(optimizer["gtol"]),
            "energy_backend": str(optimizer["energy_backend"]),
            "gradient_backend": str(optimizer["gradient_backend"]),
            "optimizer_exact_reference_used": False,
            "restarts": 4,
            "optimizer_seed": int(seed.optimizer_seed),
            "best_restart": int(result.best_restart) + 1,
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "restart_summaries": list(summaries),
        },
        architecture_binding={
            "policy": policy,
            "architecture": architecture.to_json_dict(),
            "architecture_id": architecture.architecture_id,
            "generator_pool_ordered_atom_contract_sha256": pool.ordered_atom_contract_sha256,
            "generator_pool": pool.to_json_dict(),
            "runtime_parameter_count": int(runtime.runtime_parameter_count),
            "parameterization_sha256": _canonical_sha(layout_payload),
            "theta_sha256": hashlib.sha256(np.asarray(theta, dtype="<f8").tobytes()).hexdigest(),
            "exact_reference": dict(exact_meta),
            "conditioning_search_used": False,
            "snapshot_conditioning_used": False,
            "optimizer_exact_reference_used": False,
        },
    )


def materialize_seed(
    order: HHSeedWorkOrder,
    *,
    repository_root: str | Path,
    compute: Callable[[HHSeedWorkOrder], SeedComputation] = compute_seed,
) -> SeedQualificationReceipt:
    """Run and atomically seal one canonical seed artifact and receipt."""

    root = Path(repository_root).resolve()
    seed = validate_canonical_seed_request(order.request)
    if seed != order.seed or dict(order.architecture) != architecture_policy():
        raise ProspectiveExecutionError("HH seed work order changed before execution")
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
    "HHSeedWorkOrder",
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
