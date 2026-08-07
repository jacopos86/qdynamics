"""Canonical conventional Append-ADAPT facade for Paper I.

Append-ADAPT shares the RA package's candidate inventories, compiled state
execution, accounting convention, and protocol serialization.  Its selector
and accepted-refit convention are deliberately independent: every controller
round ranks the complete executable pool by the absolute commutator gradient,
admits exactly the largest entry at the append position, and refits the full
ansatz with Powell in its native logical-shared coordinates.  Nothing in this
module calls the RA Phase-I/II/III funnel or constructs RA's supported-FS
accepted-refit chart.

The public request contains no study-policy knobs.  Bundle materialization may
construct a fully resolved protocol with policy labels for matched-study
provenance, but those labels do not change the conventional Append selector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.contracts.problem import ResolvedProblemContext
from pipelines.static_adapt.ra_adapt.adapters import (
    MacroCandidateAdapter,
    SinglePauliWordCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    APPEND_ADAPT_PROTOCOL_SCHEMA,
    APPEND_ADAPT_RESULT_SCHEMA,
    APPEND_CONVENTIONAL_SELECTOR_ID,
    APPEND_CONVENTIONAL_SELECTOR_SCOPE,
    AppendAdaptRequest,
    AppendAdaptResult,
    BundleProtocolMaterializationAuthority,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    CandidateInventoryLineageReceipt,
    CandidateLineageReceipt,
    EXACT_ORDERED_INSERTION_CHART,
    FULL_ENLARGED_ACCEPTED_REFIT,
    LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART,
    NATIVE_REFIT_CHART,
    PhaseIIIMultiplierContract,
    PolicyEchoReceipt,
    PROJECTED_GENERALIZED_SOLVER,
    RESOURCE_WEIGHTING_ALL_PHASE,
    ResolvedRAAdaptProtocol,
    SOURCE_GRAM_NO_OVERLAP_TRUST,
    canonical_json_bytes,
    canonical_sha256,
    require_protocol_materialization_authority,
)
from pipelines.static_adapt.ra_adapt.engine import (
    RA_ADAPT_COMPILE_IDENTITY,
    RA_ADAPT_ESTIMATOR_ACCOUNTING,
)
from pipelines.static_adapt.ra_adapt.pools import (
    CandidateInventory,
    CandidateRecord,
    build_candidate_inventory_lineage_receipt,
)
from pipelines.static_adapt.ra_adapt.replay_evidence import (
    build_append_controller_replay_evidence,
    build_signed_append_prefix_checkpoint,
)
from pipelines.static_adapt.ra_adapt.exact_reference_isolation import (
    build_study1_exact_reference_isolation_receipt,
    is_study1_protocol,
)
from pipelines.static_adapt.estimator_call_ledger import (
    CALL_KEY_SCHEMA_V2,
    EstimatorCallKey,
    EstimatorCallLedger,
    projective_state_fingerprint,
)
from pipelines.static_adapt.geometry_fingerprints import (
    compiled_hamiltonian_fingerprint,
)
from pipelines.static_adapt.numerical_physical_integrity import (
    build_append_numerical_physical_integrity,
    numerical_physical_integrity_from_mapping,
)
from pipelines.static_adapt.sr_snake.contracts import (
    FreshStart,
    ResolvedProblemReceipt,
)


APPEND_ADAPT_ALGORITHM_ID = "paper_i_append_adapt_v1"
APPEND_EXECUTION_SCHEMA = "paper_i_append_adapt_execution_v1"
APPEND_SOURCE_LOCK_SCHEMA = "paper_i_append_adapt_source_locks_v1"
APPEND_NATIVE_REFIT_RECEIPT_SCHEMA = (
    "paper_i_append_native_accepted_refit_receipt_v1"
)
APPEND_SELECTOR_SOURCE_ID = (
    "generic_static_full_meta_largest_absolute_commutator_gradient_v1"
)

_REQUIRED_SOURCE_LOCK_KEYS = (
    "problem_request_sha256",
    "append_module_sha256",
    "selector_module_sha256",
    "accepted_refit_module_sha256",
    "pool_module_sha256",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_lock_receipts(
    problem: ResolvedProblemContext,
) -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    static_dir = package_dir.parent
    pipelines_dir = static_dir.parent
    return {
        "problem_request_sha256": (
            ResolvedProblemReceipt.from_problem(
                problem
            ).problem_request_sha256
        ),
        "append_module_sha256": _sha256_file(Path(__file__).resolve()),
        "selector_module_sha256": _sha256_file(
            pipelines_dir
            / "exact_bench"
            / "generic_static_adapt_variants.py"
        ),
        "accepted_refit_module_sha256": _sha256_file(
            static_dir / "accepted_refit.py"
        ),
        "pool_module_sha256": _sha256_file(package_dir / "pools.py"),
    }


def _require_paper_i_problem(problem: ResolvedProblemContext) -> None:
    if not isinstance(problem, ResolvedProblemContext):
        raise TypeError("problem must be a ResolvedProblemContext.")
    if (
        str(problem.family_key).strip().lower() != "hh"
        or int(problem.request.num_sites) != 2
    ):
        raise ValueError(
            "The canonical Paper-I Append-ADAPT facade is locked to the "
            "Hubbard--Holstein L=2 problem."
        )


def _append_inventories(
    problem: ResolvedProblemContext,
    request: AppendAdaptRequest,
) -> tuple[CandidateInventory, CandidateInventory]:
    """Resolve the common parent supply and Append's global executable pool."""

    adapter = request.adapter
    parent = adapter.parent_inventory(problem)
    if isinstance(adapter, MacroCandidateAdapter):
        executable = adapter.executable_pool(problem)
    elif isinstance(adapter, SinglePauliWordCandidateAdapter):
        # This is the defining RA/Append representation difference.  Append
        # constructs the complete guarded child pool before any gradient scan.
        executable = adapter.global_executable_pool(problem)
    else:
        raise TypeError(
            "Append-ADAPT accepts only the canonical MacroCandidateAdapter "
            "or SinglePauliWordCandidateAdapter."
        )
    if not executable.candidates:
        raise ValueError("Append-ADAPT executable pool cannot be empty.")
    return parent, executable


def _ordinary_bundle_digest() -> str:
    return canonical_sha256(
        {
            "schema": "ordinary_append_adapt_facade_authority_v1",
            "bundle_id": "ordinary_append_facade_v1",
        }
    )


def build_resolved_append_protocol(
    problem: ResolvedProblemContext,
    request: AppendAdaptRequest,
    *,
    materialization_authority: (
        BundleProtocolMaterializationAuthority | None
    ) = None,
) -> ResolvedRAAdaptProtocol:
    """Materialize one immutable, source-locked Append protocol.

    Study-policy labels are obtained only from the private typed bundle
    authority.  They are intentionally absent from
    :class:`AppendAdaptRequest` and never route the conventional selector
    through RA's staged response model.
    """

    _require_paper_i_problem(problem)
    if not isinstance(request, AppendAdaptRequest):
        raise TypeError("request must be AppendAdaptRequest.")
    if materialization_authority is None:
        active_gradient_policy = ACTIVE_GRADIENT_MEASURED
        resource_weighting_scope = RESOURCE_WEIGHTING_ALL_PHASE
        bundle_id = "ordinary_append_facade_v1"
        bundle_manifest_sha256 = _ordinary_bundle_digest()
        supplied_locks: dict[str, str] = {}
        materialization_receipt = None
    else:
        if not isinstance(
            materialization_authority,
            BundleProtocolMaterializationAuthority,
        ):
            raise TypeError(
                "materialization_authority must be minted by "
                "ra_adapt.bundles."
            )
        materialization_receipt = materialization_authority.receipt
        if (
            materialization_receipt.protocol_schema
            != APPEND_ADAPT_PROTOCOL_SCHEMA
            or materialization_receipt.algorithm_id
            != APPEND_ADAPT_ALGORITHM_ID
            or materialization_receipt.selector_identity
            != APPEND_CONVENTIONAL_SELECTOR_ID
            or materialization_receipt.candidate_representation
            != str(request.adapter.candidate_representation_id)
        ):
            raise ValueError(
                "Append materialization authority does not match the "
                "request."
            )
        active_gradient_policy = (
            materialization_receipt.active_gradient_policy
        )
        resource_weighting_scope = (
            materialization_receipt.resource_weighting_scope
        )
        bundle_id = materialization_receipt.bundle_id
        bundle_manifest_sha256 = (
            materialization_receipt.bundle_manifest_sha256
        )
        supplied_locks = dict(
            materialization_authority.source_lock_refs
        )
    if not isinstance(request.execution.resume, FreshStart):
        raise ValueError(
            "Canonical Append-ADAPT currently accepts only a fresh-start "
            "execution policy; resume requires an authenticated Append "
            "checkpoint contract."
        )

    parent, executable = _append_inventories(problem, request)
    candidate_inventory_lineage = (
        build_candidate_inventory_lineage_receipt(executable)
    )
    expected_locks = _source_lock_receipts(problem)
    supplied_locks = {
        str(key): str(value) for key, value in supplied_locks.items()
    }
    for key, expected in expected_locks.items():
        supplied = supplied_locks.get(key)
        if supplied is not None and supplied != expected:
            raise ValueError(
                f"Append source lock {key!r} disagrees with the active "
                "checkout."
            )
    resolved_locks = {**supplied_locks, **expected_locks}
    horizon = int(request.execution.stop.maximum_controller_rounds)
    payload: dict[str, Any] = {
        "schema": APPEND_ADAPT_PROTOCOL_SCHEMA,
        "algorithm_id": APPEND_ADAPT_ALGORITHM_ID,
        "candidate_representation": str(
            request.adapter.candidate_representation_id
        ),
        "adapter_id": str(request.adapter.adapter_id),
        "selector_identity": APPEND_CONVENTIONAL_SELECTOR_ID,
        "selector_scope": APPEND_CONVENTIONAL_SELECTOR_SCOPE,
        "active_gradient_policy": str(active_gradient_policy),
        "resource_weighting_scope": str(resource_weighting_scope),
        "derivative_chart_id": EXACT_ORDERED_INSERTION_CHART,
        # These shared protocol fields describe the common infrastructure.
        # The conventional selector does not run a Phase-III/trust solve.
        "trust_policy_id": SOURCE_GRAM_NO_OVERLAP_TRUST,
        "phase3_solver_id": PROJECTED_GENERALIZED_SOLVER,
        "phase3_multiplier_contract": PhaseIIIMultiplierContract(),
        "accepted_refit_scope": FULL_ENLARGED_ACCEPTED_REFIT,
        "accepted_refit_coordinate_chart": NATIVE_REFIT_CHART,
        "accepted_refit_base_chart_policy": (
            LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART
        ),
        "problem": ResolvedProblemReceipt.from_problem(problem),
        "parent_inventory": parent.receipt,
        "executable_pool": executable.receipt,
        "optimizer": "powell",
        "optimizer_maxiter": 200,
        "stopping_rule": request.execution.stop.to_dict(),
        "horizon": horizon,
        "seeds": {"adapt": 7, "transpiler": 7},
        "estimator_accounting_convention": RA_ADAPT_ESTIMATOR_ACCOUNTING,
        "compile_identity": dict(RA_ADAPT_COMPILE_IDENTITY),
        "lineage_authority": {
            "source_lock_schema": APPEND_SOURCE_LOCK_SCHEMA,
            "selector_source_id": APPEND_SELECTOR_SOURCE_ID,
            "selector_source_sha256": expected_locks[
                "selector_module_sha256"
            ],
            "selector_rule": (
                "single largest absolute ADAPT commutator gradient per "
                "iteration over the complete executable pool"
            ),
            "selection_with_replacement": True,
            "append_position_only": True,
            "ra_staged_funnel_invoked": False,
            "selector_scope": APPEND_CONVENTIONAL_SELECTOR_SCOPE,
            "candidate_inventory_lineage": (
                candidate_inventory_lineage.authority_binding()
            ),
        },
        "source_locks": resolved_locks,
        "bundle_id": str(bundle_id),
        "bundle_manifest_sha256": str(bundle_manifest_sha256),
        "execution_authorized": False,
        "request": request,
    }
    if materialization_receipt is not None:
        payload["bundle_materialization"] = materialization_receipt
    return ResolvedRAAdaptProtocol(
        **payload,
        sha256=canonical_sha256(payload),
        _materialization_authority=materialization_authority,
    )


def _same_pool_identity(
    observed: CandidateInventory,
    expected: Any,
) -> bool:
    return observed.receipt == expected


def _validate_resolved_append_protocol(
    problem: ResolvedProblemContext,
    protocol: ResolvedRAAdaptProtocol,
) -> tuple[
    AppendAdaptRequest,
    CandidateInventory,
    CandidateInventory,
    CandidateInventoryLineageReceipt,
]:
    _require_paper_i_problem(problem)
    if protocol.schema != APPEND_ADAPT_PROTOCOL_SCHEMA:
        raise ValueError("run_append_adapt requires an Append protocol.")
    if protocol.algorithm_id != APPEND_ADAPT_ALGORITHM_ID:
        raise ValueError("Unknown canonical Append algorithm identity.")
    if protocol.selector_identity != APPEND_CONVENTIONAL_SELECTOR_ID:
        raise ValueError("Append protocol selector identity drifted.")
    if protocol.selector_scope != APPEND_CONVENTIONAL_SELECTOR_SCOPE:
        raise ValueError(
            "Resolved Append protocol is not scoped to the conventional "
            "no-Phase-III/no-trust selector."
        )
    if protocol.accepted_refit_coordinate_chart != NATIVE_REFIT_CHART:
        raise ValueError(
            "Resolved Append protocol requires the native accepted-refit chart."
        )
    if (
        protocol.accepted_refit_base_chart_policy
        != LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART
    ):
        raise ValueError(
            "Resolved Append protocol requires the logical-shared native "
            "accepted-refit coordinates."
        )
    request = protocol.request
    if not isinstance(request, AppendAdaptRequest):
        raise TypeError("Resolved Append protocol lost its typed request.")
    if int(request.execution.stop.maximum_controller_rounds) != int(
        protocol.horizon
    ):
        raise ValueError("Resolved Append protocol horizon drifted.")
    if protocol.adapter_id != str(request.adapter.adapter_id):
        raise ValueError("Resolved Append adapter identity drifted.")
    if protocol.candidate_representation != str(
        request.adapter.candidate_representation_id
    ):
        raise ValueError("Resolved Append representation drifted.")
    if protocol.problem != ResolvedProblemReceipt.from_problem(problem):
        raise ValueError("Resolved Append problem receipt drifted.")
    if not isinstance(request.execution.resume, FreshStart):
        raise ValueError(
            "Resolved Append protocol has no authenticated resume contract."
        )

    expected_locks = _source_lock_receipts(problem)
    for key in _REQUIRED_SOURCE_LOCK_KEYS:
        if str(protocol.source_locks.get(key, "")) != str(
            expected_locks[key]
        ):
            raise ValueError(
                f"Resolved Append source lock {key!r} drifted."
            )
    parent, executable = _append_inventories(problem, request)
    if not _same_pool_identity(parent, protocol.parent_inventory):
        raise ValueError("Resolved Append parent inventory drifted.")
    if not _same_pool_identity(executable, protocol.executable_pool):
        raise ValueError("Resolved Append executable pool drifted.")
    candidate_inventory_lineage = (
        build_candidate_inventory_lineage_receipt(executable)
    )
    if protocol.lineage_authority.get(
        "candidate_inventory_lineage"
    ) != candidate_inventory_lineage.authority_binding():
        raise ValueError(
            "Resolved Append candidate inventory lineage drifted."
        )
    return request, parent, executable, candidate_inventory_lineage


def _select_largest_absolute_commutator_gradient(
    scored: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Return the deterministic conventional Append winner.

    Sorting here makes the extracted rule independently testable and fixes the
    tie rule to candidate label without importing any RA ranking surface.
    """

    rows = [dict(row) for row in scored]
    if not rows:
        return None
    rows.sort(
        key=lambda row: (
            -float(
                row.get(
                    "abs_gradient_decision",
                    row.get("abs_gradient", 0.0),
                )
            ),
            str(row.get("label", "")),
        )
    )
    return rows[0]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Append result serialization forbids nonfinite values.")
    return value


def _native_accepted_refit_receipt(
    *,
    base_chart_policy: str,
    origin_logical_theta: np.ndarray,
    origin_runtime_theta: np.ndarray,
    final_logical_theta: np.ndarray,
    final_runtime_theta: np.ndarray,
    final_energy: float,
) -> dict[str, Any]:
    """Describe one ordinary full-ansatz Powell refit without FS queries."""

    origin_logical = np.asarray(
        origin_logical_theta, dtype=float
    ).reshape(-1)
    origin_runtime = np.asarray(
        origin_runtime_theta, dtype=float
    ).reshape(-1)
    final_logical = np.asarray(final_logical_theta, dtype=float).reshape(-1)
    final_runtime = np.asarray(final_runtime_theta, dtype=float).reshape(-1)
    if int(origin_logical.size) < 1:
        raise ValueError("Append native refit requires an admitted coordinate.")
    if final_logical.shape != origin_logical.shape:
        raise ValueError(
            "Append native refit changed the logical optimizer dimension."
        )
    if final_runtime.shape != origin_runtime.shape:
        raise ValueError(
            "Append native refit changed the runtime parameter dimension."
        )
    if float(origin_logical[-1]) != 0.0:
        raise ValueError(
            "Append native refit requires the admitted coordinate to start at zero."
        )
    for name, values in (
        ("origin_logical_theta", origin_logical),
        ("origin_runtime_theta", origin_runtime),
        ("final_logical_theta", final_logical),
        ("final_runtime_theta", final_runtime),
    ):
        if not bool(np.all(np.isfinite(values))):
            raise ValueError(f"Append native refit {name} must be finite.")
    if not math.isfinite(float(final_energy)):
        raise ValueError("Append native refit energy must be finite.")

    payload = {
        "schema": APPEND_NATIVE_REFIT_RECEIPT_SCHEMA,
        "scope": FULL_ENLARGED_ACCEPTED_REFIT,
        "coordinate_chart": NATIVE_REFIT_CHART,
        "base_chart_policy": str(base_chart_policy),
        "base_chart_applied": None,
        "optimizer_coordinate_mode": "logical_shared",
        "native_ansatz_coordinate_contract": (
            "one_logical_parameter_per_admitted_generator_v1"
        ),
        "origin_kind": "inherited_parameters_plus_zero_admitted_coordinate_v1",
        "origin_logical_theta": [
            float(value) for value in origin_logical.tolist()
        ],
        "origin_runtime_theta": [
            float(value) for value in origin_runtime.tolist()
        ],
        "logical_parameter_count": int(origin_logical.size),
        "runtime_parameter_count": int(origin_runtime.size),
        "optimizer_parameter_count": int(origin_logical.size),
        "admitted_logical_coordinate_index": int(origin_logical.size - 1),
        "admitted_coordinate_initialized_to_zero": True,
        "chart_fixed_within_powell_invocation": True,
        "supported_fs_chart_constructed": False,
        "whitening_performed": False,
        "metric_backend_evaluation_performed": False,
        "chart_origin_hamiltonian_acquisition_count": 0,
        "chart_origin_gradient_acquisition_count": 0,
        "chart_origin_metric_acquisition_count": 0,
        "final_logical_theta": [
            float(value) for value in final_logical.tolist()
        ],
        "final_runtime_theta": [
            float(value) for value in final_runtime.tolist()
        ],
        "final_energy": float(final_energy),
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _write_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(_jsonable(payload)))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class _AppendExecutableCandidate:
    label: str
    polynomial: Any
    support: tuple[int, ...]
    pauli_labels_exyz: tuple[str, ...]
    execution_mode: str


def _polynomial_labels_and_support(
    polynomial: Any,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    labels: list[str] = []
    support: set[int] = set()
    for term in polynomial.return_polynomial():
        if abs(complex(term.p_coeff)) <= 1.0e-12:
            continue
        label = str(term.pw2strng()).lower()
        if not label:
            continue
        labels.append(label)
        qubit_count = int(getattr(term, "N", len(label)))
        support.update(
            int(qubit_count - 1 - index)
            for index, character in enumerate(label)
            if character != "e"
        )
    return tuple(labels), tuple(sorted(support))


def _append_candidate(
    record: CandidateRecord,
) -> _AppendExecutableCandidate:
    """Project one canonical pool record onto the Append execution kernel."""

    labels, support = _polynomial_labels_and_support(
        record.term.polynomial
    )
    if not labels or not support:
        raise ValueError(
            f"Append candidate {record.label!r} has no executable support."
        )
    return _AppendExecutableCandidate(
        label=str(record.label),
        polynomial=record.term.polynomial,
        support=tuple(int(value) for value in support),
        pauli_labels_exyz=tuple(str(value).lower() for value in labels),
        execution_mode=str(record.execution_mode),
    )


def _compile_append_resources(
    *,
    problem: ResolvedProblemContext,
    protocol: ResolvedRAAdaptProtocol,
    selected_records: Sequence[CandidateRecord],
    reference_state: np.ndarray,
    source_kind: str,
) -> dict[str, Any]:
    """Compile one accepted Append prefix under the shared Table-I contract."""

    from importlib import metadata as importlib_metadata

    from pipelines.exact_bench.table_i_qiskit_resource_compile import (
        TABLE_I_QISKIT_COMPILE_CONVENTION,
        TableICompileUnavailable,
        TableIQiskitCompileConfig,
        compile_table_i_ansatz_terms,
    )

    compile_identity = dict(protocol.compile_identity)
    if str(compile_identity.get("policy")) != (
        TABLE_I_QISKIT_COMPILE_CONVENTION
    ):
        raise RuntimeError("Append protocol carries a foreign compile identity.")
    compile_config = TableIQiskitCompileConfig(
        basis_gates=tuple(
            str(value) for value in compile_identity["basis_gates"]
        ),
        optimization_level=int(compile_identity["optimization_level"]),
        seed_transpiler=(
            None
            if compile_identity.get("transpiler_seed") is None
            else int(compile_identity["transpiler_seed"])
        ),
        include_reference_state=bool(
            compile_identity["reference_preparation_included"]
        ),
    )
    try:
        return dict(
            compile_table_i_ansatz_terms(
                ops=tuple(record.term for record in selected_records),
                num_qubits=int(problem.layout.total_qubits),
                reference_state=np.asarray(
                    reference_state, dtype=complex
                ).reshape(-1),
                source_kind=str(source_kind),
                config=compile_config,
            )
        )
    except TableICompileUnavailable as exc:
        try:
            qiskit_version = importlib_metadata.version("qiskit")
        except importlib_metadata.PackageNotFoundError:
            qiskit_version = None
        return {
            "compiled_circuit_stats_status": str(exc.status),
            "compiled_resource_source_kind": str(source_kind),
            "compiled_resource_qiskit_validated": False,
            "compiled_basis_gates": list(compile_config.basis_gates),
            "compile_convention": TABLE_I_QISKIT_COMPILE_CONVENTION,
            "qiskit_version": qiskit_version,
            "qiskit_transpile_optimization_level": int(
                compile_config.optimization_level
            ),
            "qiskit_transpile_seed": compile_config.seed_transpiler,
            "compiled_circuit_scope": (
                "ansatz_circuit_including_reference_state"
                if compile_config.include_reference_state
                else "ansatz_circuit_no_reference_state"
            ),
            "reason": str(exc.reason),
        }


def _execute_conventional_append(
    problem: ResolvedProblemContext,
    protocol: ResolvedRAAdaptProtocol,
    executable: CandidateInventory,
) -> Mapping[str, Any]:
    """Execute the conventional selector on the protocol-locked pool.

    The conventional gradient rule is extracted from the retained generic
    comparator.  The pool is supplied by the canonical RA package and every
    accepted ansatz is refit by ordinary Powell in its native logical-shared
    ansatz coordinates.
    """

    from scipy.optimize import minimize

    from src.quantum.ansatz_parameterization import (
        build_parameter_layout,
        expand_legacy_logical_theta,
    )
    from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
    from src.quantum.compiled_polynomial import (
        adapt_commutator_grad_from_hpsi,
        apply_compiled_polynomial,
        compile_polynomial_action,
    )

    request = protocol.request
    if not isinstance(request, AppendAdaptRequest):
        raise TypeError("Append execution requires its typed request.")
    records = tuple(executable.candidates)
    append_pool = tuple(_append_candidate(record) for record in records)
    by_label = {
        str(candidate.label): (record, candidate)
        for record, candidate in zip(records, append_pool, strict=True)
    }
    if len(by_label) != len(records):
        raise RuntimeError("Append executable pool labels must be unique.")

    pauli_action_cache: dict[str, Any] = {}
    compiled_pool = tuple(
        (
            record,
            candidate,
            compile_polynomial_action(
                candidate.polynomial,
                tol=1.0e-12,
                pauli_action_cache=pauli_action_cache,
            ),
        )
        for record, candidate in zip(records, append_pool, strict=True)
    )
    h_compiled = compile_polynomial_action(
        problem.hamiltonian,
        tol=1.0e-12,
        pauli_action_cache=pauli_action_cache,
    )
    estimator_call_ledger = EstimatorCallLedger()
    estimator_hamiltonian_fingerprint = compiled_hamiltonian_fingerprint(
        h_compiled
    )
    estimator_backend_fingerprint = "compiled_exact_statevector_v1"
    estimator_precision_contract = "complex128_float64_deterministic_v1"
    direct_hamiltonian_action_count = 0
    direct_gradient_evaluation_count = 0

    def _record_estimator_call(
        *,
        state: np.ndarray,
        component: str,
        consumer_scope: str,
        primitive_kind: str,
        observable_or_formula_identity: str,
        operand_identity: str | None = None,
        symmetric_pair: tuple[str, str] | None = None,
    ) -> None:
        key_kwargs: dict[str, Any] = {
            "projective_state_fingerprint": projective_state_fingerprint(
                np.asarray(state, dtype=complex).reshape(-1)
            ),
            "hamiltonian_fingerprint": (
                estimator_hamiltonian_fingerprint
            ),
            "backend_fingerprint": estimator_backend_fingerprint,
            "precision_contract": estimator_precision_contract,
            "primitive_kind": str(primitive_kind),
            "observable_or_formula_identity": str(
                observable_or_formula_identity
            ),
        }
        if operand_identity is not None or symmetric_pair is not None:
            key_kwargs.update(
                schema=CALL_KEY_SCHEMA_V2,
                operand_identity=operand_identity,
                symmetric_pair=symmetric_pair,
            )
        estimator_call_ledger.record_call(
            EstimatorCallKey(**key_kwargs),
            component=str(component),
            consumer_scope=str(consumer_scope),
        )

    def _apply_hamiltonian(state: np.ndarray) -> np.ndarray:
        nonlocal direct_hamiltonian_action_count
        direct_hamiltonian_action_count += 1
        return np.asarray(
            apply_compiled_polynomial(state, h_compiled),
            dtype=complex,
        ).reshape(-1)

    psi_ref = np.asarray(
        problem.reference_state.build_state(), dtype=complex
    ).reshape(-1)
    expected_dimension = 1 << int(problem.layout.total_qubits)
    if int(psi_ref.size) != expected_dimension:
        raise ValueError(
            "Append reference-state dimension does not match the problem "
            "layout."
        )
    norm = float(np.linalg.norm(psi_ref))
    if not math.isfinite(norm) or not np.isclose(
        norm, 1.0, rtol=1.0e-10, atol=1.0e-12
    ):
        raise ValueError("Append reference state must be normalized.")

    selected_records: list[CandidateRecord] = []
    logical_theta = np.zeros(0, dtype=float)
    runtime_theta = np.zeros(0, dtype=float)
    history: list[dict[str, Any]] = []
    requested_resource_rounds = frozenset(
        int(value)
        for value in (request.observation.resource_rounds or ())
    )
    compiled_resources_by_round: list[dict[str, Any]] = []
    stop_reason = "maximum_controller_rounds"
    final_state = psi_ref.copy()
    final_energy: float | None = None

    def _write_checkpoint(
        *,
        status: str,
        force: bool,
        estimator_accounting: Mapping[str, Any] | None = None,
        terminal_resources: Mapping[str, Any] | None = None,
        controller_replay_evidence: Mapping[str, Any] | None = None,
    ) -> None:
        observation = request.observation.checkpoint
        if observation is None:
            return
        completed_rounds = int(len(history))
        if (
            not force
            and completed_rounds % int(observation.every_controller_rounds)
            != 0
        ):
            return
        keep = int(observation.keep_history_tail)
        history_tail = (
            []
            if keep == 0
            else [dict(row) for row in history[-keep:]]
        )
        checkpoint_payload: dict[str, Any] = {
            "schema": "paper_i_append_adapt_checkpoint_v1",
            "algorithm_id": protocol.algorithm_id,
            "protocol_sha256": protocol.sha256,
            "selector_identity": APPEND_CONVENTIONAL_SELECTOR_ID,
            "selector_scope": APPEND_CONVENTIONAL_SELECTOR_SCOPE,
            "status": str(status),
            "controller_rounds_completed": completed_rounds,
            "history_total_count": completed_rounds,
            "history_tail": history_tail,
            "keep_history_tail": keep,
            "every_controller_rounds": int(
                observation.every_controller_rounds
            ),
            "accepted_operator_labels": [
                str(record.label) for record in selected_records
            ],
            "accepted_generator_identities": [
                str(record.generator_identity)
                for record in selected_records
            ],
            "logical_theta": [
                float(value)
                for value in np.asarray(
                    logical_theta, dtype=float
                ).reshape(-1)
            ],
            "runtime_theta": [
                float(value)
                for value in np.asarray(
                    runtime_theta, dtype=float
                ).reshape(-1)
            ],
            "final_energy": (
                None if final_energy is None else float(final_energy)
            ),
            "stop_reason": str(stop_reason),
            "requested_resource_rounds": sorted(
                requested_resource_rounds
            ),
            "compiled_resources_by_round": [
                dict(row) for row in compiled_resources_by_round
            ],
            "estimator_call_ledger": (
                estimator_call_ledger.to_payload()
            ),
        }
        if estimator_accounting is not None:
            checkpoint_payload["estimator_accounting"] = dict(
                estimator_accounting
            )
        if terminal_resources is not None:
            checkpoint_payload["compiled_resources"] = dict(
                terminal_resources
            )
        if controller_replay_evidence is not None:
            checkpoint_payload["controller_replay_evidence"] = dict(
                controller_replay_evidence
            )
        checkpoint_payload["sha256"] = canonical_sha256(
            checkpoint_payload
        )
        _write_canonical_json(observation.path, checkpoint_payload)

    for controller_round in range(1, int(protocol.horizon) + 1):
        if selected_records:
            selected_terms = [record.term for record in selected_records]
            current_layout = build_parameter_layout(
                selected_terms,
                ignore_identity=True,
                coefficient_tolerance=1.0e-12,
                sort_terms=True,
            )
            current_executor = CompiledAnsatzExecutor(
                selected_terms,
                coefficient_tolerance=1.0e-12,
                ignore_identity=True,
                sort_terms=True,
                pauli_action_cache=pauli_action_cache,
                parameterization_mode="per_pauli_term",
                parameterization_layout=current_layout,
            )
            runtime_theta = np.asarray(
                expand_legacy_logical_theta(
                    logical_theta, current_layout
                ),
                dtype=float,
            )
            current_state = np.asarray(
                current_executor.prepare_state(runtime_theta, psi_ref),
                dtype=complex,
            ).reshape(-1)
        else:
            current_state = psi_ref.copy()
        h_current = _apply_hamiltonian(current_state)
        energy_before = float(np.real(np.vdot(current_state, h_current)))
        _record_estimator_call(
            state=current_state,
            component="N_H_outer",
            consumer_scope=(
                f"append_round_{controller_round}:outer_energy"
            ),
            primitive_kind="hamiltonian_expectation",
            observable_or_formula_identity="hamiltonian_expectation_v1",
        )
        scored: list[dict[str, Any]] = []
        for record, candidate, compiled in compiled_pool:
            direct_gradient_evaluation_count += 1
            candidate_state = np.asarray(
                apply_compiled_polynomial(current_state, compiled),
                dtype=complex,
            ).reshape(-1)
            gradient = float(
                adapt_commutator_grad_from_hpsi(
                    h_current, candidate_state
                )
            )
            _record_estimator_call(
                state=current_state,
                component="N_grad",
                consumer_scope=(
                    f"append_round_{controller_round}:global_pool_gradient"
                ),
                primitive_kind="coordinate_gradient",
                observable_or_formula_identity=(
                    "adapt_commutator_gradient_v1"
                ),
                operand_identity=(
                    "append_candidate_tangent_v1:"
                    + str(record.generator_identity)
                    + ":position="
                    + str(len(selected_records))
                ),
            )
            scored.append(
                {
                    "label": str(candidate.label),
                    "gradient": gradient,
                    "abs_gradient": float(abs(gradient)),
                    "support": list(candidate.support),
                    "pauli_labels_exyz": list(
                        candidate.pauli_labels_exyz
                    ),
                }
            )
        winner = _select_largest_absolute_commutator_gradient(scored)
        if winner is None:
            stop_reason = "executable_pool_empty"
            final_state = current_state
            final_energy = energy_before
            break
        label = str(winner["label"])
        record, _executable_candidate = by_label[label]
        append_position = len(selected_records)
        geometry = request.adapter.candidate_geometry(
            record,
            int(append_position),
        )
        geometry_payload = (
            geometry.as_dict()
            if hasattr(geometry, "as_dict")
            else _jsonable(geometry)
        )
        selected_lineage = CandidateLineageReceipt(
            representation_id=str(record.representation_id),
            candidate_label=str(record.label),
            generator_identity=str(record.generator_identity),
            parent_identities=tuple(
                str(value) for value in record.parent_identities
            ),
            insertion_position=int(append_position),
            candidate_manifest_sha256=canonical_sha256(
                record.manifest_row()
            ),
        )
        selected_records.append(record)
        logical_theta = np.concatenate(
            [logical_theta, np.zeros(1, dtype=float)]
        )
        origin_logical_theta = np.asarray(
            logical_theta, dtype=float
        ).reshape(-1)
        selected_terms = [item.term for item in selected_records]
        layout = build_parameter_layout(
            selected_terms,
            ignore_identity=True,
            coefficient_tolerance=1.0e-12,
            sort_terms=True,
        )
        if int(origin_logical_theta.size) != int(
            layout.logical_parameter_count
        ):
            raise RuntimeError(
                "Append native refit lost one-logical-coordinate-per-"
                "admitted-generator closure."
            )
        origin_runtime_theta = np.asarray(
            expand_legacy_logical_theta(origin_logical_theta, layout),
            dtype=float,
        ).reshape(-1)
        logical_executor = CompiledAnsatzExecutor(
            selected_terms,
            coefficient_tolerance=1.0e-12,
            ignore_identity=True,
            sort_terms=True,
            pauli_action_cache=pauli_action_cache,
            parameterization_mode="logical_shared",
            parameterization_layout=layout,
            enable_prefix_state_cache=True,
        )
        objective_calls = 0
        objective_cache: dict[tuple[int, bytes], tuple[float, np.ndarray]] = {}

        def _runtime_key(theta_value: np.ndarray) -> tuple[int, bytes]:
            array = np.ascontiguousarray(
                np.asarray(theta_value, dtype=np.float64).reshape(-1)
            )
            return int(array.size), array.tobytes()

        def native_objective(theta_value: np.ndarray) -> float:
            nonlocal objective_calls
            theta_array = np.asarray(theta_value, dtype=float).reshape(-1)
            if int(theta_array.size) != int(
                layout.logical_parameter_count
            ):
                raise ValueError(
                    "Append native Powell objective requires one coordinate "
                    "per admitted generator."
                )
            key = _runtime_key(theta_array)
            cached = objective_cache.get(key)
            if cached is not None:
                return float(cached[0])
            objective_calls += 1
            state = np.asarray(
                logical_executor.prepare_state(theta_array, psi_ref),
                dtype=complex,
            ).reshape(-1)
            h_state = _apply_hamiltonian(state)
            energy = float(np.real(np.vdot(state, h_state)))
            _record_estimator_call(
                state=state,
                component="N_H_refit",
                consumer_scope=(
                    f"append_round_{controller_round}:accepted_refit_powell"
                ),
                primitive_kind="hamiltonian_expectation",
                observable_or_formula_identity=(
                    "hamiltonian_expectation_v1"
                ),
            )
            objective_cache[key] = (energy, state)
            return energy

        optimizer_result = minimize(
            native_objective,
            origin_logical_theta,
            method="Powell",
            options={
                "maxiter": int(protocol.optimizer_maxiter),
                "xtol": 1.0e-5,
                "ftol": 1.0e-12,
            },
        )
        optimizer_logical_theta = np.asarray(
            getattr(optimizer_result, "x", origin_logical_theta),
            dtype=float,
        ).reshape(-1)
        if int(optimizer_logical_theta.size) != int(
            layout.logical_parameter_count
        ):
            raise RuntimeError(
                "Append native Powell returned a foreign optimizer "
                "dimension."
            )
        logical_theta = optimizer_logical_theta
        runtime_theta = np.asarray(
            expand_legacy_logical_theta(logical_theta, layout),
            dtype=float,
        ).reshape(-1)
        final_energy = float(native_objective(logical_theta))
        final_state = objective_cache[
            _runtime_key(logical_theta)
        ][1].copy()
        refit = _native_accepted_refit_receipt(
            base_chart_policy=protocol.accepted_refit_base_chart_policy,
            origin_logical_theta=origin_logical_theta,
            origin_runtime_theta=origin_runtime_theta,
            final_logical_theta=logical_theta,
            final_runtime_theta=runtime_theta,
            final_energy=final_energy,
        )
        accepted_prefix_checkpoint = build_signed_append_prefix_checkpoint(
            protocol=protocol,
            controller_round=int(controller_round),
            accepted_operator_labels=tuple(
                str(item.label) for item in selected_records
            ),
            accepted_generator_identities=tuple(
                str(item.generator_identity) for item in selected_records
            ),
            logical_parameters=tuple(
                float(value) for value in logical_theta.tolist()
            ),
            runtime_parameters=tuple(
                float(value) for value in runtime_theta.tolist()
            ),
            projective_state_fingerprint=projective_state_fingerprint(
                final_state
            ),
            accepted_energy=float(final_energy),
            accepted_refit=refit,
            estimator_prefix=(
                estimator_call_ledger.closed_occurrence_prefix_summary()
            ),
        )
        history.append(
            {
                "controller_round": int(controller_round),
                "selector_identity": APPEND_CONVENTIONAL_SELECTOR_ID,
                "selector_scope": APPEND_CONVENTIONAL_SELECTOR_SCOPE,
                "selector_source_id": APPEND_SELECTOR_SOURCE_ID,
                "candidate_count_scored": int(len(scored)),
                "selected_label": label,
                "selected_generator_identity": str(
                    record.generator_identity
                ),
                "selected_parent_identities": list(
                    record.parent_identities
                ),
                "selected_candidate_lineage": (
                    selected_lineage.to_dict()
                ),
                "selected_abs_commutator_gradient": float(
                    winner.get("abs_gradient", 0.0)
                ),
                "insertion_position": int(append_position),
                "candidate_geometry": geometry_payload,
                "energy_before": float(energy_before),
                "energy_after": float(final_energy),
                "accepted_refit": dict(refit),
                "active_prefix_checkpoint": accepted_prefix_checkpoint,
                "optimizer": {
                    "method": "scipy.optimize.minimize:Powell",
                    "success": bool(
                        getattr(optimizer_result, "success", False)
                    ),
                    "message": str(
                        getattr(optimizer_result, "message", "")
                    ),
                    "nit": (
                        None
                        if getattr(optimizer_result, "nit", None) is None
                        else int(optimizer_result.nit)
                    ),
                    "nfev": int(objective_calls),
                },
                "ra_staged_funnel_invoked": False,
            }
        )
        if int(controller_round) in requested_resource_rounds:
            prefix_resources = _compile_append_resources(
                problem=problem,
                protocol=protocol,
                selected_records=tuple(selected_records),
                reference_state=psi_ref,
                source_kind=(
                    "paper_i_append_adapt_accepted_prefix_v1"
                ),
            )
            compiled_resources_by_round.append(
                {
                    "controller_round": int(controller_round),
                    "accepted_prefix_length": int(
                        len(selected_records)
                    ),
                    "compiled_resources": prefix_resources,
                }
            )
        _write_checkpoint(status="in_progress", force=False)
        exact_stop = request.execution.stop.exact_ed_target
        if exact_stop is not None and abs(
            final_energy - float(exact_stop.energy)
        ) <= float(exact_stop.absolute_tolerance):
            stop_reason = "exact_ed_target"
            break

    if final_energy is None:
        raise RuntimeError(
            "Append execution completed without one recorded controller-round "
            "Hamiltonian evaluation."
        )
    occurrence_summary = estimator_call_ledger.occurrence_summary()
    closed_prefix = estimator_call_ledger.closed_occurrence_prefix_summary()
    components = {
        key: int(occurrence_summary[key])
        for key in ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
    }
    s_alg = int(sum(components.values()))
    if int(occurrence_summary["S_alg"]) != s_alg:
        raise RuntimeError("Append estimator occurrence accounting did not close.")
    prefix_s_alg = int(
        closed_prefix["cumulative_executed_queries"]["S_alg"]
    )
    if prefix_s_alg != s_alg:
        raise RuntimeError(
            "Append estimator ledger prefix disagrees with occurrence "
            "accounting."
        )
    executed_occurrences = {
        "N_H_outer_and_refit": int(direct_hamiltonian_action_count),
        "N_grad": int(direct_gradient_evaluation_count),
        "N_metric": 0,
    }
    if (
        executed_occurrences["N_H_outer_and_refit"]
        != components["N_H_outer"] + components["N_H_refit"]
        or executed_occurrences["N_grad"] != components["N_grad"]
        or executed_occurrences["N_metric"] != components["N_metric"]
    ):
        raise RuntimeError(
            "Append executed primitive instrumentation disagrees with the "
            "estimator ledger."
        )
    accounting = {
        "schema": "paper_i_append_estimator_accounting_v2",
        "convention": protocol.estimator_accounting_convention,
        "components": components,
        **components,
        "S_alg": s_alg,
        "closed_occurrence_reconciliation": True,
        "occurrence_summary": occurrence_summary,
        "closed_occurrence_prefix": closed_prefix,
        "executed_occurrence_instrumentation": {
            **executed_occurrences,
            "closed_against_estimator_ledger": True,
        },
    }
    estimator_ledger_payload = estimator_call_ledger.to_payload()
    numerical_physical_integrity = (
        build_append_numerical_physical_integrity(
            problem=problem,
            final_state=final_state,
            history=history,
            logical_parameters=logical_theta,
            runtime_parameters=runtime_theta,
            final_energy=float(final_energy),
        )
    )
    controller_replay_evidence = (
        build_append_controller_replay_evidence(
            protocol=protocol,
            history=history,
            estimator_ledger=estimator_ledger_payload,
            estimator_accounting=accounting,
        )
    )
    resources = _compile_append_resources(
        problem=problem,
        protocol=protocol,
        selected_records=tuple(selected_records),
        reference_state=psi_ref,
        source_kind="paper_i_append_adapt_terminal_ansatz_v1",
    )
    _write_checkpoint(
        status="completed",
        force=True,
        estimator_accounting=accounting,
        terminal_resources=resources,
        controller_replay_evidence=controller_replay_evidence,
    )
    return {
        "schema": APPEND_EXECUTION_SCHEMA,
        "algorithm_id": protocol.algorithm_id,
        "protocol_sha256": protocol.sha256,
        "selector_identity": APPEND_CONVENTIONAL_SELECTOR_ID,
        "selector_scope": APPEND_CONVENTIONAL_SELECTOR_SCOPE,
        "selector_source_id": APPEND_SELECTOR_SOURCE_ID,
        "candidate_representation": protocol.candidate_representation,
        "parent_inventory": protocol.parent_inventory.to_dict(),
        "executable_pool": protocol.executable_pool.to_dict(),
        "global_pool_constructed_before_gradient_selection": bool(
            protocol.candidate_representation
            == CANDIDATE_REPRESENTATION_SINGLE_PAULI
        ),
        "selection_with_replacement": True,
        "append_position_only": True,
        "candidate_geometry_chart": EXACT_ORDERED_INSERTION_CHART,
        "ra_staged_funnel_invoked": False,
        "phase3_solver_invoked": False,
        "trust_transaction_invoked": False,
        "accepted_operator_labels": [
            str(record.label) for record in selected_records
        ],
        "accepted_generator_identities": [
            str(record.generator_identity) for record in selected_records
        ],
        "logical_theta": [
            float(value) for value in logical_theta.tolist()
        ],
        "runtime_theta": [
            float(value) for value in runtime_theta.tolist()
        ],
        "final_energy": float(final_energy),
        "final_state_serialized": False,
        "controller_rounds_completed": int(len(history)),
        "stop_reason": str(stop_reason),
        "history": history,
        "estimator_accounting": accounting,
        "estimator_call_ledger": estimator_ledger_payload,
        "numerical_physical_integrity": (
            numerical_physical_integrity.to_dict()
        ),
        "controller_replay_evidence": controller_replay_evidence,
        "compile_identity": dict(protocol.compile_identity),
        "compiled_resources": resources,
        "resource_observation": {
            "requested_resource_rounds": sorted(
                requested_resource_rounds
            ),
            "materialized_resource_rounds": [
                int(row["controller_round"])
                for row in compiled_resources_by_round
            ],
            "unmaterialized_resource_rounds": sorted(
                requested_resource_rounds.difference(
                    int(row["controller_round"])
                    for row in compiled_resources_by_round
                )
            ),
        },
        "compiled_resources_by_round": compiled_resources_by_round,
        "accepted_refit_scope": protocol.accepted_refit_scope,
        "accepted_refit_coordinate_chart": (
            protocol.accepted_refit_coordinate_chart
        ),
        "policy_echo": {
            "selector_scope": APPEND_CONVENTIONAL_SELECTOR_SCOPE,
            "active_gradient_policy": protocol.active_gradient_policy,
            "resource_weighting_scope": (
                protocol.resource_weighting_scope
            ),
            "conventional_selector_consumes_ra_study_policies": False,
        },
    }


def run_append_adapt(
    problem: ResolvedProblemContext,
    request: AppendAdaptRequest | ResolvedRAAdaptProtocol | None = None,
) -> AppendAdaptResult:
    """Execute one canonical Paper-I conventional Append-ADAPT request."""

    _require_paper_i_problem(problem)
    if request is None:
        public_request = AppendAdaptRequest()
        protocol = build_resolved_append_protocol(
            problem, public_request
        )
        (
            _request,
            parent,
            executable,
            candidate_inventory_lineage,
        ) = _validate_resolved_append_protocol(problem, protocol)
    elif isinstance(request, AppendAdaptRequest):
        public_request = request
        protocol = build_resolved_append_protocol(
            problem, public_request
        )
        (
            _request,
            parent,
            executable,
            candidate_inventory_lineage,
        ) = _validate_resolved_append_protocol(problem, protocol)
    elif isinstance(request, ResolvedRAAdaptProtocol):
        protocol = request
        require_protocol_materialization_authority(
            protocol,
            ordinary_algorithm_id=APPEND_ADAPT_ALGORITHM_ID,
            ordinary_bundle_id="ordinary_append_facade_v1",
            ordinary_bundle_manifest_sha256=_ordinary_bundle_digest(),
        )
        (
            public_request,
            parent,
            executable,
            candidate_inventory_lineage,
        ) = (
            _validate_resolved_append_protocol(problem, protocol)
        )
    else:
        raise TypeError(
            "request must be AppendAdaptRequest, a bundle-resolved Append "
            "protocol, or None."
        )

    require_protocol_materialization_authority(
        protocol,
        ordinary_algorithm_id=APPEND_ADAPT_ALGORITHM_ID,
        ordinary_bundle_id="ordinary_append_facade_v1",
        ordinary_bundle_manifest_sha256=_ordinary_bundle_digest(),
    )
    result_payload = dict(
        _execute_conventional_append(problem, protocol, executable)
    )
    if (
        result_payload.get("selector_identity")
        != APPEND_CONVENTIONAL_SELECTOR_ID
    ):
        raise RuntimeError("Append executor returned a foreign selector.")
    if (
        result_payload.get("selector_scope")
        != APPEND_CONVENTIONAL_SELECTOR_SCOPE
    ):
        raise RuntimeError(
            "Append executor returned a foreign or missing selector scope."
        )
    if result_payload.get("ra_staged_funnel_invoked") is not False:
        raise RuntimeError(
            "Append executor did not attest separation from the RA funnel."
        )
    if str(result_payload.get("protocol_sha256")) != protocol.sha256:
        raise RuntimeError("Append executor protocol identity drifted.")
    policy = PolicyEchoReceipt(
        active_gradient_policy=protocol.active_gradient_policy,
        resource_weighting_scope=protocol.resource_weighting_scope,
        active_gradient_indices_acquired=(),
        active_gradient_charge=0,
    )
    from pipelines.reporting.paper_i_append_run_summary import (
        summarize_paper_i_append_run,
    )

    paper_i_summary = summarize_paper_i_append_run(
        protocol=protocol,
        selector_identity=APPEND_CONVENTIONAL_SELECTOR_ID,
        result_payload=result_payload,
    )
    numerical_physical_integrity = (
        numerical_physical_integrity_from_mapping(
            result_payload.get("numerical_physical_integrity")
        )
    )
    study1_g8 = (
        build_study1_exact_reference_isolation_receipt(
            protocol=protocol,
            method="append_adapt",
            finalized_controller_rounds=int(
                result_payload["controller_rounds_completed"]
            ),
            exact_same_cutoff_energy=float(
                problem.exact_target.resolve_energy()
            ),
        )
        if is_study1_protocol(protocol)
        else None
    )
    scientific_receipts = {
        "selector_identity": APPEND_CONVENTIONAL_SELECTOR_ID,
        "selector_scope": APPEND_CONVENTIONAL_SELECTOR_SCOPE,
        "selector_source_id": APPEND_SELECTOR_SOURCE_ID,
        "selector_rule": (
            "largest_absolute_commutator_gradient_then_append_v1"
        ),
        "selection_with_replacement": True,
        "ra_staged_funnel_invoked": False,
        "phase3_solver_invoked": False,
        "trust_transaction_invoked": False,
        "candidate_representation": protocol.candidate_representation,
        "candidate_inventory_lineage": (
            candidate_inventory_lineage.to_dict()
        ),
        "parent_inventory_sha256": (
            protocol.parent_inventory.ordered_pool_sha256
        ),
        "executable_pool_sha256": (
            protocol.executable_pool.ordered_pool_sha256
        ),
        "candidate_geometry_chart": EXACT_ORDERED_INSERTION_CHART,
        "accepted_refit_scope": FULL_ENLARGED_ACCEPTED_REFIT,
        "accepted_refit_coordinate_chart": (
            protocol.accepted_refit_coordinate_chart
        ),
        "accepted_refit_base_chart_policy": (
            protocol.accepted_refit_base_chart_policy
        ),
        "compile_identity": dict(protocol.compile_identity),
        "estimator_accounting_convention": (
            protocol.estimator_accounting_convention
        ),
        "policy": policy.to_dict(),
        "paper_i_append_run_summary": paper_i_summary.to_dict(),
        "paper_i_append_run_summary_sha256": canonical_sha256(
            paper_i_summary
        ),
        "controller_replay_evidence": result_payload[
            "controller_replay_evidence"
        ],
        "controller_replay_evidence_sha256": (
            result_payload["controller_replay_evidence"]["sha256"]
        ),
        **(
            {
                "study1_g8_exact_reference_isolation": (
                    study1_g8.to_dict()
                )
            }
            if study1_g8 is not None
            else {}
        ),
        "numerical_physical_integrity": (
            numerical_physical_integrity.to_dict()
        ),
        "numerical_physical_integrity_sha256": canonical_sha256(
            numerical_physical_integrity
        ),
    }
    result = AppendAdaptResult(
        schema=APPEND_ADAPT_RESULT_SCHEMA,
        protocol=protocol,
        selector_identity=APPEND_CONVENTIONAL_SELECTOR_ID,
        parent_inventory=parent.receipt,
        executable_pool=executable.receipt,
        policy=policy,
        result_payload=result_payload,
        paper_i_summary=paper_i_summary,
        numerical_physical_integrity=numerical_physical_integrity,
        scientific_receipts=scientific_receipts,
    )
    if public_request.observation.estimator_ledger is not None:
        ledger = result_payload.get("estimator_call_ledger")
        if not isinstance(ledger, Mapping):
            raise RuntimeError(
                "Append execution omitted its estimator occurrence ledger."
            )
        _write_canonical_json(
            public_request.observation.estimator_ledger.path,
            dict(ledger),
        )
    return result


__all__ = [
    "APPEND_ADAPT_ALGORITHM_ID",
    "APPEND_EXECUTION_SCHEMA",
    "APPEND_SELECTOR_SOURCE_ID",
    "APPEND_SOURCE_LOCK_SCHEMA",
    "build_resolved_append_protocol",
    "run_append_adapt",
]
