"""Method runners: strict intake -> repository methods, with per-method receipts.

Every runner returns a ``BridgeMethodReceipt`` whose ``status`` is one of

    cpu_verified         the method executed on a CPU backend and its checks passed
    gpu_verified         the method executed on an accelerator (``cuda``/``mps``)
                         whose device was read back from a live tensor, and its
                         CPU-parity checks passed
    skipped_unsupported  the requested backend/device/dtype cannot be exercised here
    blocked              the repository route refuses this problem family (reason recorded)
    failed               a check failed (details recorded)

A GPU run that silently fell back to CPU can never be reported as
``gpu_verified``: the status is derived from ``BackendReport.exercised_device``.

Lane status for a molecular electronic problem at this commit:

* Paper III (QSE): ``pipelines.qse_spectra.core.compute_qse_spectra`` is
  family-neutral and is executed directly (CPU).  The same pencil is assembled
  on the bridge backend for CPU/GPU parity.
* Paper II (AP-McLachlan): ``run_append_mclachlan_trajectory`` is executed
  directly on a ``ScaffoldRuntimeInput`` built from the resolved context (CPU
  only; the route's geometry is dense numpy).  Real reference + real H + an
  odd-Y (UCC) support has zero McLachlan force, so the candidate pool must
  contain even-Y generators; the repository's ``full_meta`` molecular pool
  supplies them (Hamiltonian blocks and words).
* Paper I (RA-ADAPT/SNAKE): the typed facade ``run_ra_adapt`` rejects every
  family except HH L in {2,3}, named Hubbard applications and the H2O
  linear-FD family (``pipelines/static_adapt/ra_adapt/engine.py``).  The bridge
  resolves the repository's molecular pool plan and evaluates the ADAPT
  selection gradients on the backend; the full route stays ``blocked`` until a
  named lane-owned application admits the family.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.chemistry_bridge import BRIDGE_SCHEMA_VERSION
from pipelines.chemistry_bridge.array_backend import (
    BackendReport,
    BackendUnavailable,
    NumpyBackend,
    TorchBackend,
    parity_tolerances,
    resolve_backend,
)
from pipelines.chemistry_bridge.polynomial_action import (
    BoundPolynomialAction,
    BridgePolynomialAction,
    adapt_commutator_gradients,
    fixed_sector_basis_indices,
    out_of_sector_norm,
    qse_pencil,
    sector_matrix,
    sector_spectrum,
)
from pipelines.chemistry_bridge.qdynamics_export import declaration_from_mapping, read_bridge_manifest
from pipelines.contracts.problem import ProblemRequest, ResolvedProblemContext
from pipelines.static_adapt.builders.molecular_diagnostic_intake import (
    MolecularDiagnosticDeclaration,
    resolve_molecular_diagnostic_intake,
)
from src.quantum.compiled_polynomial import (
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial,
    compile_polynomial_action,
    energy_via_one_apply,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial


STATUS_CPU_VERIFIED = "cpu_verified"
STATUS_GPU_VERIFIED = "gpu_verified"
STATUS_SKIPPED_UNSUPPORTED = "skipped_unsupported"
STATUS_BLOCKED = "blocked"
STATUS_FAILED = "failed"

RECEIPT_SCHEMA = "doron_bridge_method_receipt_v1"
MOLECULAR_FAMILY = "molecular_restricted_closed_shell"

Backend = NumpyBackend | TorchBackend


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, complex):
        return {"re": float(value.real), "im": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def verified_status(report: BackendReport, *, checks_passed: bool) -> str:
    """Derive the status from the live device report, never from the request."""

    if not report.available:
        return STATUS_SKIPPED_UNSUPPORTED
    if report.exercised_device != report.requested_device:
        # A request that landed somewhere else is a fallback; it is never
        # counted, not even as a CPU result.
        return STATUS_FAILED
    if not checks_passed:
        return STATUS_FAILED
    if report.exercised_device in {"cuda", "mps"}:
        return STATUS_GPU_VERIFIED
    if report.exercised_device == "cpu":
        return STATUS_CPU_VERIFIED
    return STATUS_FAILED


@dataclass
class BridgeMethodReceipt:
    method: str
    lane: str
    route: str
    status: str
    backend: dict[str, Any]
    inputs: dict[str, Any]
    results: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    schema: str = RECEIPT_SCHEMA
    created_utc: str = field(default_factory=_now)

    def to_json_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class BridgeContext:
    """A strict-intake context plus the identities a receipt must carry."""

    context: ResolvedProblemContext
    intake_receipt: dict[str, Any]
    declaration: MolecularDiagnosticDeclaration
    manifest: dict[str, Any]
    provenance: dict[str, Any]
    data_role: str

    @property
    def nq(self) -> int:
        return int(self.context.layout.total_qubits)

    @property
    def n_spatial(self) -> int:
        return int(self.context.request.num_sites)

    @property
    def num_particles(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.context.default_num_particles)  # type: ignore[return-value]

    def input_identity(self) -> dict[str, Any]:
        receipt = self.intake_receipt
        return {
            "problem_json_path": self.manifest.get("problem_json_path"),
            "raw_sha256": receipt.get("raw_sha256"),
            "tensor_sha256": receipt.get("tensor_sha256"),
            "semantic_sha256": receipt.get("semantic_sha256"),
            "operator_sha256": receipt.get("operator_sha256"),
            "diagnostic_atol": receipt.get("diagnostic_atol"),
            "data_role": self.data_role,
            "source": self.provenance.get("source", {}),
            "producer": self.provenance.get("producer", {}),
            "reduction": self.provenance.get("reduction", {}),
            "n_qubits": self.nq,
            "num_particles": list(self.num_particles),
        }


def resolve_bridge_context(manifest_path: str | Path, *, atol: float = 1.0e-12) -> BridgeContext:
    """Run the strict intake on a bridge problem document named by its manifest."""

    manifest = read_bridge_manifest(manifest_path)
    problem_json = Path(manifest["problem_json_path"])
    document = json.loads(problem_json.read_text(encoding="utf-8"))
    provenance = dict(document.get("provenance", {}))
    declaration = declaration_from_mapping(manifest["declaration"])
    request = ProblemRequest(
        problem_key=MOLECULAR_FAMILY,
        num_sites=int(manifest["n_spatial_orbitals"]),
        t=0.0, u=0.0, dv=0.0, omega0=0.0, g_ep=0.0,
        n_ph_max=0, boson_encoding="binary", ordering="blocked", boundary="open",
        include_zero_point=True, molecular_problem_json=str(problem_json),
    )
    context = resolve_molecular_diagnostic_intake(
        request,
        expected_raw_sha256=str(manifest["raw_sha256"]),
        declaration=declaration,
        atol=float(atol),
    )
    receipt = dict(context.runtime_data["molecular_diagnostic_intake_receipt"])
    source_kind = str(provenance.get("source", {}).get("payload_kind", "unknown"))
    # The schema says how the bytes are laid out, not who produced them: a file
    # in the producer's schema counts as Doron's data only once its sha256 is
    # pinned to a delivery he made.  No such pin exists yet.
    data_role = "ele_h5_file_input_origin_unverified" if source_kind == "qdynamics_matrix_elements_ele_h5" else "synthetic_or_non_file_input"
    return BridgeContext(context=context, intake_receipt=receipt, declaration=declaration, manifest=manifest, provenance=provenance, data_role=data_role)


def _backend_or_receipt(spec: str, *, method: str, lane: str, route: str, inputs: dict[str, Any]) -> tuple[Backend | None, BridgeMethodReceipt | None]:
    try:
        backend = resolve_backend(spec)
    except BackendUnavailable as exc:
        from pipelines.chemistry_bridge.array_backend import backend_availability

        report = backend_availability(spec)
        return None, BridgeMethodReceipt(
            method=method, lane=lane, route=route, status=STATUS_SKIPPED_UNSUPPORTED,
            backend=report.to_json_dict(), inputs=inputs,
            notes=[f"backend {spec!r} unavailable: {exc}"],
        )
    return backend, None


def hf_statevector(bridge: BridgeContext) -> np.ndarray:
    return np.asarray(bridge.context.reference_state.build_state(), dtype=np.complex128).reshape(-1)


def hamiltonian_action(bridge: BridgeContext, *, compile_tol: float = 1.0e-15) -> BridgePolynomialAction:
    return BridgePolynomialAction.from_polynomial(bridge.context.hamiltonian, compile_tol=compile_tol)


# ----------------------------------------------------------------------------
# Mapping witness
# ----------------------------------------------------------------------------


def run_mapping_witness(bridge: BridgeContext, backend_spec: str = "numpy:float64", *, seed: int = 1) -> BridgeMethodReceipt:
    """Check the mapped operator against tensor-level witnesses on the backend.

    * HF expectation of the mapped operator equals the intake's input-tensor
      determinant energy (the JW mapping, spin blocking and scalar insertion
      agree with the tensors).
    * The action is Hermitian on random states.
    * The action preserves the (n_alpha, n_beta) sector.

    The HF witness also proves the scalar slot is inserted exactly once: the
    mapped expectation equals the tensor determinant energy that already
    contains it.  (The identity Pauli coefficient itself is not the scalar: the
    Jordan-Wigner images of number operators contribute identity terms too.)
    """

    started = time.perf_counter()
    inputs = bridge.input_identity()
    backend, skipped = _backend_or_receipt(backend_spec, method="mapping_witness", lane="shared_chemistry_input", route="bridge_polynomial_action", inputs=inputs)
    if skipped is not None:
        return skipped
    report = backend.report()
    tol = parity_tolerances(report.real_dtype)
    action = hamiltonian_action(bridge)
    bound = action.bind(backend)
    psi_hf = hf_statevector(bridge)
    witness = float(bridge.intake_receipt["checks"]["input_tensor_determinant_energy"])
    hf_energy = bound.expectation_numpy(psi_hf)
    rng = np.random.default_rng(int(seed))
    dim = 1 << bridge.nq
    phi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    chi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    phi /= np.linalg.norm(phi)
    chi /= np.linalg.norm(chi)
    h_phi = bound.apply_numpy(phi)
    h_chi = bound.apply_numpy(chi)
    hermiticity = abs(np.vdot(chi, h_phi) - np.vdot(h_chi, phi))
    basis = fixed_sector_basis_indices(bridge.n_spatial, bridge.num_particles)
    block, leakage = sector_matrix(bound, basis)
    scalar_slot = float(bridge.declaration.nuclear_scalar_hartree + bridge.declaration.frozen_core_scalar_hartree)
    identity = complex(action.identity_coefficient)
    checks = {
        "hf_expectation_minus_tensor_witness_abs": abs(hf_energy.real - witness),
        "hf_expectation_imag_abs": abs(hf_energy.imag),
        "hermiticity_defect_abs": float(hermiticity),
        "sector_leakage_max_norm": float(leakage),
        "identity_coefficient_imag_abs": abs(identity.imag),
        "sector_dimension": int(basis.size),
        "term_count": int(action.term_count),
    }
    passed = (
        checks["hf_expectation_minus_tensor_witness_abs"] <= tol["scalar_abs"] * max(1.0, abs(witness))
        and checks["hf_expectation_imag_abs"] <= tol["scalar_abs"]
        and checks["hermiticity_defect_abs"] <= tol["scalar_abs"] * 10
        and checks["sector_leakage_max_norm"] <= tol["vector_abs"] * 10
        and checks["identity_coefficient_imag_abs"] <= 1.0e-12
    )
    return BridgeMethodReceipt(
        method="mapping_witness", lane="shared_chemistry_input", route="bridge_polynomial_action",
        status=verified_status(report, checks_passed=passed), backend=report.to_json_dict(), inputs=inputs,
        results={"hf_expectation": hf_energy.real, "tensor_witness_determinant_energy": witness, "action_sha256": action.action_sha256, "scalar_slot": scalar_slot, "identity_pauli_coefficient": identity.real},
        checks=checks, elapsed_s=time.perf_counter() - started,
        notes=["tensor witness is the intake's input-tensor determinant energy, not an SCF stationarity proof"],
    )


# ----------------------------------------------------------------------------
# Exact reference (fixed-sector dense diagonalization assembled on the backend)
# ----------------------------------------------------------------------------


def run_exact_reference(bridge: BridgeContext, backend_spec: str = "numpy:float64", *, n_roots: int = 4, compare_repository_target: bool = True) -> BridgeMethodReceipt:
    started = time.perf_counter()
    inputs = bridge.input_identity()
    backend, skipped = _backend_or_receipt(backend_spec, method="exact_sector_reference", lane="shared_chemistry_reference", route="bridge_sector_matrix_eigvalsh", inputs=inputs)
    if skipped is not None:
        return skipped
    report = backend.report()
    tol = parity_tolerances(report.real_dtype)
    bound = hamiltonian_action(bridge).bind(backend)
    basis = fixed_sector_basis_indices(bridge.n_spatial, bridge.num_particles)
    block, leakage = sector_matrix(bound, basis)
    spectrum = sector_spectrum(block, hermitian_atol=tol["matrix_abs"] * 10)
    roots = [float(v) for v in spectrum[: int(n_roots)]]
    checks: dict[str, Any] = {"sector_leakage_max_norm": float(leakage), "sector_dimension": int(basis.size)}
    results: dict[str, Any] = {"ground_energy": roots[0], "low_roots": roots, "eigensolver": "numpy.linalg.eigvalsh on CPU over the backend-assembled block"}
    passed = leakage <= tol["vector_abs"] * 10
    if compare_repository_target:
        repository_energy = float(bridge.context.exact_target.resolve_energy())
        checks["repository_exact_target_minus_bridge_abs"] = abs(repository_energy - roots[0])
        results["repository_exact_target"] = repository_energy
        passed = passed and checks["repository_exact_target_minus_bridge_abs"] <= tol["scalar_abs"] * max(1.0, abs(repository_energy))
    return BridgeMethodReceipt(
        method="exact_sector_reference", lane="shared_chemistry_reference", route="bridge_sector_matrix_eigvalsh",
        status=verified_status(report, checks_passed=passed), backend=report.to_json_dict(), inputs=inputs,
        results=results, checks=checks, elapsed_s=time.perf_counter() - started,
        notes=["reporting input only; never an online controller input"],
    )


# ----------------------------------------------------------------------------
# Paper III: QSE
# ----------------------------------------------------------------------------


def _uccsd_pool(bridge: BridgeContext) -> list[Any]:
    from src.quantum.chemistry.molecular_uccsd import build_molecular_uccsd_pool

    return list(build_molecular_uccsd_pool(n_spatial_orbitals=bridge.n_spatial, num_particles=bridge.num_particles, ordering="blocked"))


def run_qse(
    bridge: BridgeContext,
    backend_spec: str = "numpy:float64",
    *,
    basis_kind: str = "uccsd_singles_doubles",
    include_identity: bool = True,
    reference_projection: str = "none",
    basis_vector_normalization: str = "normalized",
    n_roots_report: int = 4,
    exact_roots: Sequence[float] | None = None,
) -> BridgeMethodReceipt:
    """Execute the repository QSE core on the HF reference with a UCC-generator basis.

    The authoritative result is ``compute_qse_spectra`` (CPU numpy).  The same
    pencil is assembled on the requested backend and compared entrywise; when
    the backend is an accelerator and parity holds, the status is
    ``gpu_verified``.  If ``exact_roots`` are supplied and the basis spans the
    sector, the Ritz roots are compared with them.
    """

    from pipelines.qse_spectra.core import (
        QSEBasisVectorPolicy,
        compute_qse_spectra,
        pauli_string_basis_element,
        polynomial_basis_element,
    )

    started = time.perf_counter()
    inputs = bridge.input_identity()
    backend, skipped = _backend_or_receipt(backend_spec, method="qse", lane="paper_iii", route="pipelines.qse_spectra.core.compute_qse_spectra", inputs=inputs)
    if skipped is not None:
        return skipped
    report = backend.report()
    tol = parity_tolerances(report.real_dtype)
    pool = _uccsd_pool(bridge)
    if basis_kind == "uccsd_singles_doubles":
        chosen = pool
    elif basis_kind == "uccsd_singles":
        chosen = [term for term in pool if str(term.label).startswith("uccsd_sing(")]
    else:
        raise ValueError("basis_kind must be 'uccsd_singles_doubles' or 'uccsd_singles'")
    psi_hf = hf_statevector(bridge)
    elements = []
    bridge_actions: list[BoundPolynomialAction | None] = []
    if include_identity:
        elements.append(pauli_string_basis_element("e" * bridge.nq, nq=bridge.nq, name="identity"))
        bridge_actions.append(None)
    for term in chosen:
        elements.append(polynomial_basis_element(term.polynomial, name=str(term.label)))
        bridge_actions.append(BridgePolynomialAction.from_polynomial(term.polynomial).bind(backend))
    policy = QSEBasisVectorPolicy(reference_projection=reference_projection, basis_vector_normalization=basis_vector_normalization)
    result = compute_qse_spectra(bridge.context.hamiltonian, psi_hf, elements, basis_vector_policy=policy)
    core_overlap = np.asarray(result.matrices.overlap, dtype=np.complex128)
    core_hamiltonian = np.asarray(result.matrices.hamiltonian, dtype=np.complex128)
    h_bound = hamiltonian_action(bridge).bind(backend)
    pencil = qse_pencil(h_bound, bridge_actions, psi_hf, reference_projection=reference_projection, basis_vector_normalization=basis_vector_normalization)
    overlap_diff = float(np.max(np.abs(pencil.overlap - core_overlap))) if core_overlap.size else 0.0
    hamiltonian_diff = float(np.max(np.abs(pencil.hamiltonian - core_hamiltonian))) if core_hamiltonian.size else 0.0
    roots = [float(v) for v in np.asarray(result.eigenvalues, dtype=float)[: int(n_roots_report)]]
    checks: dict[str, Any] = {
        "backend_overlap_minus_core_max_abs": overlap_diff,
        "backend_hamiltonian_minus_core_max_abs": hamiltonian_diff,
        "reference_energy_backend_minus_core_abs": abs(pencil.reference_energy - float(result.matrices.reference_energy)),
        "core_overlap_hermitian_residual_raw": float(result.matrices.overlap_hermitian_residual_max_abs_raw),
        "core_hamiltonian_hermitian_residual_raw": float(result.matrices.hamiltonian_hermitian_residual_max_abs_raw),
        "retained_rank": int(result.retained_rank),
        "discarded_rank": int(result.discarded_rank),
        "basis_size": int(len(elements)),
    }
    passed = overlap_diff <= tol["matrix_abs"] and hamiltonian_diff <= tol["matrix_abs"] * max(1.0, float(np.max(np.abs(core_hamiltonian)))) and checks["reference_energy_backend_minus_core_abs"] <= tol["scalar_abs"]
    results: dict[str, Any] = {
        "ritz_roots": roots,
        "reference_energy": float(result.matrices.reference_energy),
        "solver_status": str(result.solver_status),
        "basis_labels": [str(element.name) for element in elements],
        "policy": {"reference_projection": reference_projection, "basis_vector_normalization": basis_vector_normalization},
    }
    sector_dimension = int(fixed_sector_basis_indices(bridge.n_spatial, bridge.num_particles).size)
    results["sector_dimension"] = sector_dimension
    results["basis_spans_sector"] = bool(int(result.retained_rank) == sector_dimension)
    if exact_roots is not None:
        exact = np.asarray(list(exact_roots), dtype=float)
        ritz = np.asarray(result.eigenvalues, dtype=float)
        count = min(exact.size, ritz.size)
        diff = float(np.max(np.abs(np.sort(ritz)[:count] - np.sort(exact)[:count]))) if count else 0.0
        checks["ritz_minus_exact_sector_roots_max_abs"] = diff
        results["exact_roots_compared"] = int(count)
        # Ritz values are upper bounds; equality with the exact sector roots is
        # expected only when the retained basis spans the whole sector.
        if results["basis_spans_sector"]:
            passed = passed and diff <= tol["scalar_abs"] * max(1.0, float(np.max(np.abs(exact[:count]))) if count else 1.0) * 10
        ritz_sorted = np.sort(ritz)[:count]
        exact_sorted = np.sort(exact)[:count]
        checks["ritz_below_exact_max"] = float(np.max(exact_sorted - ritz_sorted)) if count else 0.0
    return BridgeMethodReceipt(
        method="qse", lane="paper_iii", route="pipelines.qse_spectra.core.compute_qse_spectra",
        status=verified_status(report, checks_passed=passed), backend=report.to_json_dict(), inputs=inputs,
        results=results, checks=checks, elapsed_s=time.perf_counter() - started,
        notes=[
            "authoritative pencil and roots come from the repository core on CPU; the backend pencil is the parity witness",
            "the UCC-generator basis is a bridge choice, not Doron data; R (roots reported), sector and observables remain user decisions",
        ],
    )


# ----------------------------------------------------------------------------
# Paper I: static ADAPT selection gradients + facade gate record
# ----------------------------------------------------------------------------


def resolve_molecular_pool(bridge: BridgeContext, pool_key: str = "full_meta") -> tuple[list[Any], dict[str, Any]]:
    from pipelines.static_adapt.builders.pool_resolution import resolve_pool_plan

    plan = resolve_pool_plan(
        resolved_problem=bridge.context,
        continuation_mode="phase3_v1",
        adapt_pool=str(pool_key),
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        phase3_symmetry_mitigation_mode="verify_only",
    )
    meta = {"pool_key": str(plan.pool_key), "method_name": str(plan.method_name), "pool_size": int(len(plan.pool)), "qpb": int(plan.qpb)}
    return list(plan.pool), meta


def facade_admission_record(bridge: BridgeContext) -> dict[str, Any]:
    """Call the typed RA facade once to record its actual admission verdict."""

    try:
        from pipelines.static_adapt.ra_adapt.engine import run_ra_adapt
    except Exception as exc:  # pragma: no cover - import environment
        return {"admitted": None, "error": f"facade import failed: {type(exc).__name__}: {exc}"}
    try:
        run_ra_adapt(bridge.context)
    except ValueError as exc:
        return {"admitted": False, "facade": "pipelines.static_adapt.ra_adapt.engine.run_ra_adapt", "message": str(exc)}
    except Exception as exc:
        return {"admitted": None, "facade": "pipelines.static_adapt.ra_adapt.engine.run_ra_adapt", "message": f"{type(exc).__name__}: {exc}"}
    return {"admitted": True, "facade": "pipelines.static_adapt.ra_adapt.engine.run_ra_adapt", "message": "facade accepted the molecular context"}


def run_static_adapt_screening(bridge: BridgeContext, backend_spec: str = "numpy:float64", *, pool_key: str = "full_meta", top_k: int = 8, record_facade_gate: bool = True) -> BridgeMethodReceipt:
    """Resolve the repository molecular pool and evaluate ADAPT selection gradients.

    Gradients ``g_k = 2 Im <H psi| A_k psi>`` at the HF reference are computed on
    the backend and compared with the repository's numpy formula.  The full
    RA-ADAPT route is recorded as ``blocked`` while the typed facade rejects the
    family; this receipt never claims an RA-ADAPT run.
    """

    started = time.perf_counter()
    inputs = bridge.input_identity()
    backend, skipped = _backend_or_receipt(backend_spec, method="static_adapt_selection_gradients", lane="paper_i", route="resolve_pool_plan + adapt_commutator_gradients", inputs=inputs)
    if skipped is not None:
        return skipped
    report = backend.report()
    tol = parity_tolerances(report.real_dtype)
    pool, pool_meta = resolve_molecular_pool(bridge, pool_key)
    psi_hf = hf_statevector(bridge)
    h_bound = hamiltonian_action(bridge).bind(backend)
    generators = [BridgePolynomialAction.from_polynomial(term.polynomial).bind(backend) for term in pool]
    gradients, energy = adapt_commutator_gradients(h_bound, generators, psi_hf)
    compiled_h = compile_polynomial_action(bridge.context.hamiltonian, tol=1.0e-15)
    energy_ref, h_psi = energy_via_one_apply(psi_hf, compiled_h)
    reference = np.array([
        adapt_commutator_grad_from_hpsi(h_psi, apply_compiled_polynomial(psi_hf, compile_polynomial_action(term.polynomial, tol=1.0e-15)))
        for term in pool
    ], dtype=float)
    gradient_diff = float(np.max(np.abs(gradients - reference))) if reference.size else 0.0
    order = np.argsort(-np.abs(gradients))
    top = [{"label": str(pool[int(i)].label), "gradient": float(gradients[int(i)])} for i in order[: int(top_k)]]
    checks = {
        "backend_gradients_minus_repository_max_abs": gradient_diff,
        "energy_backend_minus_repository_abs": abs(energy - float(energy_ref)),
        "pool_size": int(len(pool)),
    }
    scale = max(1.0, float(np.max(np.abs(reference))) if reference.size else 1.0)
    passed = gradient_diff <= tol["scalar_abs"] * scale * 10 and checks["energy_backend_minus_repository_abs"] <= tol["scalar_abs"] * max(1.0, abs(energy_ref))
    results: dict[str, Any] = {"pool": pool_meta, "hf_energy": float(energy), "top_gradients": top, "max_abs_gradient": float(np.max(np.abs(gradients))) if reference.size else 0.0}
    notes = [
        "selection gradients only: the typed RA-ADAPT facade does not admit molecular_restricted_closed_shell, so no RA-ADAPT/SNAKE run is claimed",
        "the molecular pool comes from the repository's resolve_pool_plan (phase3_v1), not from Doron data",
    ]
    status = verified_status(report, checks_passed=passed)
    if record_facade_gate:
        gate = facade_admission_record(bridge)
        results["ra_adapt_facade_gate"] = gate
        if gate.get("admitted") is False:
            results["full_route_status"] = STATUS_BLOCKED
    return BridgeMethodReceipt(
        method="static_adapt_selection_gradients", lane="paper_i", route="resolve_pool_plan + adapt_commutator_gradients",
        status=status, backend=report.to_json_dict(), inputs=inputs, results=results, checks=checks,
        elapsed_s=time.perf_counter() - started, notes=notes,
    )


# ----------------------------------------------------------------------------
# Paper II: AP-McLachlan (generalized-exchange support controller), CPU route
# ----------------------------------------------------------------------------


def _exact_sector_propagation(block: np.ndarray, basis: np.ndarray, psi0: np.ndarray, times: Sequence[float], dim: int) -> list[np.ndarray]:
    eigenvalues, vectors = np.linalg.eigh(0.5 * (block + block.conj().T))
    coefficients = vectors.conj().T @ np.asarray(psi0, dtype=np.complex128)[basis]
    states = []
    for t in times:
        sector = vectors @ (np.exp(-1j * eigenvalues * float(t)) * coefficients)
        full = np.zeros(int(dim), dtype=np.complex128)
        full[basis] = sector
        states.append(full)
    return states


def run_ap_mclachlan(
    bridge: BridgeContext,
    *,
    times: Sequence[float] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
    candidate_pool_key: str = "full_meta",
    seed_term_label: str = "ham_block::molecular_one_body",
    parameterization_mode: str = "logical_shared",
    integrator_method: str = "rk4",
    support_config_overrides: Mapping[str, Any] | None = None,
    compare_exact: bool = True,
) -> BridgeMethodReceipt:
    """Execute the repository AP-McLachlan controller from the HF reference (CPU).

    Diagnostics: energy drift along the trajectory (static H), out-of-sector
    norm of every recorded state, and fidelity against exact fixed-sector
    propagation ``exp(-iHt)|HF>`` computed from the bridge sector block.
    """

    from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
    from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
        SupportPatchControllerConfig,
        run_append_mclachlan_trajectory,
    )
    from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig
    from pipelines.time_dynamics.ap_mclachlan.hamiltonian import time_dependent_hamiltonian_from_runtime_input
    from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
    from pipelines.time_dynamics.ap_mclachlan.state import state_from_scaffold_runtime_input
    from src.quantum.ansatz_parameterization import build_parameter_layout

    started = time.perf_counter()
    inputs = bridge.input_identity()
    cpu = NumpyBackend("float64")
    report = cpu.report()
    pool, pool_meta = resolve_molecular_pool(bridge, candidate_pool_key)
    seeds = [term for term in pool if str(term.label) == str(seed_term_label)]
    if not seeds:
        raise ValueError(f"seed term {seed_term_label!r} not in pool {candidate_pool_key!r}; labels start with {sorted({str(t.label).split('(')[0] for t in pool})[:8]}")
    selected = (seeds[0],)
    candidates = tuple(term for term in pool if str(term.label) != str(seed_term_label))
    layout = build_parameter_layout(selected)
    psi_hf = hf_statevector(bridge)
    basis = fixed_sector_basis_indices(bridge.n_spatial, bridge.num_particles)
    h_bound = hamiltonian_action(bridge).bind(cpu)
    block, leakage = sector_matrix(h_bound, basis)
    exact_ground = float(sector_spectrum(block)[0])
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=bridge.context,
        psi_ref=psi_hf,
        psi_initial=psi_hf.copy(),
        base_layout=layout,
        theta_runtime=np.zeros(int(layout.runtime_parameter_count), dtype=float),
        theta_logical=np.zeros(int(layout.logical_parameter_count), dtype=float),
        structure_locked=False,
        exact_energy=exact_ground,
        selected_terms=selected,
        candidate_pool_terms=candidates,
        candidate_pool_source=CandidatePoolSource(source_kind="resolved_pool", pool_key=str(pool_meta["pool_key"]), completeness="complete"),
        provenance={"schema": BRIDGE_SCHEMA_VERSION, "intake_raw_sha256": bridge.intake_receipt.get("raw_sha256"), "intake_operator_sha256": bridge.intake_receipt.get("operator_sha256"), "seed": "hf_reference_theta_zero"},
    )
    state = state_from_scaffold_runtime_input(runtime_input, parameterization_mode=str(parameterization_mode))
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(runtime_input)
    config = SupportPatchControllerConfig(**dict(support_config_overrides or {}))
    trajectory = run_append_mclachlan_trajectory(
        state=state,
        hamiltonian=hamiltonian,
        times=[float(t) for t in times],
        inverse_policy=McLachlanInversePolicy(),
        integrator_method=str(integrator_method),
        support_patch_config=config,
        solve_repair_config=SolveRepairConfig(),
        metadata={"bridge": BRIDGE_SCHEMA_VERSION},
    )
    energies = [float(point.energy_expectation) for point in trajectory.points]
    recorded_times = [float(point.time) for point in trajectory.points]
    leak = [out_of_sector_norm(np.asarray(point.geometry.psi), basis) for point in trajectory.points]
    supports = [int(point.runtime_parameter_count) for point in trajectory.points]
    patch_kinds = [str(point.patch_decision.patch_kind) for point in trajectory.points]
    checks: dict[str, Any] = {
        "energy_drift_max_abs": float(max(energies) - min(energies)) if energies else 0.0,
        "sector_leakage_max_norm": float(max(leak)) if leak else 0.0,
        "hamiltonian_sector_leakage": float(leakage),
        "initial_energy_minus_hf_witness_abs": abs(energies[0] - float(bridge.intake_receipt["checks"]["input_tensor_determinant_energy"])) if energies else None,
        "point_count": int(len(trajectory.points)),
    }
    results: dict[str, Any] = {
        "times": recorded_times,
        "energies": energies,
        "runtime_parameter_counts": supports,
        "patch_kinds": patch_kinds,
        "pool": pool_meta,
        "seed_term": str(seed_term_label),
        "parameterization_mode": str(state.parameterization_mode),
        "integrator_method": str(trajectory.integrator_method),
        "exact_ground_energy": exact_ground,
        "final_runtime_parameter_count": int(trajectory.final_state.runtime_parameter_count),
        "support_patch_config": _json_safe(config.to_json_dict()) if hasattr(config, "to_json_dict") else {},
    }
    passed = checks["sector_leakage_max_norm"] <= 1.0e-8 and (checks["initial_energy_minus_hf_witness_abs"] or 0.0) <= 1.0e-10
    if compare_exact and trajectory.points:
        exact_states = _exact_sector_propagation(block, basis, psi_hf, recorded_times, 1 << bridge.nq)
        fidelities = [float(abs(np.vdot(exact, np.asarray(point.geometry.psi, dtype=np.complex128))) ** 2) for exact, point in zip(exact_states, trajectory.points)]
        exact_energies = [float(np.vdot(exact, h_bound.apply_numpy(exact)).real) for exact in exact_states]
        results["fidelity_vs_exact_sector_propagation"] = fidelities
        results["exact_energies"] = exact_energies
        checks["min_fidelity_vs_exact"] = float(min(fidelities))
        checks["exact_energy_drift_max_abs"] = float(max(exact_energies) - min(exact_energies))
    return BridgeMethodReceipt(
        method="ap_mclachlan", lane="paper_ii", route="pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory.run_append_mclachlan_trajectory",
        status=verified_status(report, checks_passed=passed), backend=report.to_json_dict(), inputs=inputs, results=results, checks=checks,
        elapsed_s=time.perf_counter() - started,
        notes=[
            "CPU only: the AP-McLachlan geometry evaluator is dense numpy; no GPU path exists for this route (GPU status: unsupported, not attempted)",
            "zero drive: molecular_restricted_closed_shell declares supports_driven_realtime=False; a dipole drive would be a separate bridge input",
            "candidate pool and seed term are bridge choices (repository full_meta pool), not Doron data",
            "fidelity against exact sector propagation is a diagnostic of the variational trajectory, not a certification",
        ],
    )


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------


DEFAULT_METHODS = ("mapping", "exact", "qse", "adapt", "apm")


def run_bridge(
    manifest_path: str | Path,
    *,
    backend_spec: str = "numpy:float64",
    methods: Sequence[str] = DEFAULT_METHODS,
    out_dir: str | Path | None = None,
    apm_times: Sequence[float] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
    apm_pool_key: str = "full_meta",
    apm_parameterization_mode: str = "logical_shared",
    qse_exact_compare: bool = True,
) -> dict[str, Any]:
    bridge = resolve_bridge_context(manifest_path)
    receipts: list[BridgeMethodReceipt] = []
    exact_roots: list[float] | None = None
    wanted = [str(m).strip().lower() for m in methods]
    if "mapping" in wanted:
        receipts.append(run_mapping_witness(bridge, backend_spec))
    if "exact" in wanted or (qse_exact_compare and "qse" in wanted):
        exact = run_exact_reference(bridge, backend_spec, n_roots=16)
        if "exact" in wanted:
            receipts.append(exact)
        if exact.status in {STATUS_CPU_VERIFIED, STATUS_GPU_VERIFIED}:
            exact_roots = list(exact.results["low_roots"])
    if "qse" in wanted:
        receipts.append(run_qse(bridge, backend_spec, exact_roots=exact_roots if qse_exact_compare else None))
    if "adapt" in wanted:
        receipts.append(run_static_adapt_screening(bridge, backend_spec))
    if "apm" in wanted:
        receipts.append(run_ap_mclachlan(bridge, times=apm_times, candidate_pool_key=apm_pool_key, parameterization_mode=apm_parameterization_mode))
    summary = {
        "schema": RECEIPT_SCHEMA + "_bundle",
        "created_utc": _now(),
        "backend_spec": str(backend_spec),
        "inputs": bridge.input_identity(),
        "intake_receipt": _json_safe(bridge.intake_receipt),
        "receipts": [receipt.to_json_dict() for receipt in receipts],
        "status_by_method": {receipt.method: receipt.status for receipt in receipts},
    }
    if out_dir is not None:
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "bridge_receipts.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        summary["out_path"] = str(directory / "bridge_receipts.json")
    return summary


__all__ = [
    "BridgeContext",
    "BridgeMethodReceipt",
    "DEFAULT_METHODS",
    "MOLECULAR_FAMILY",
    "RECEIPT_SCHEMA",
    "STATUS_BLOCKED",
    "STATUS_CPU_VERIFIED",
    "STATUS_FAILED",
    "STATUS_GPU_VERIFIED",
    "STATUS_SKIPPED_UNSUPPORTED",
    "facade_admission_record",
    "hamiltonian_action",
    "hf_statevector",
    "resolve_bridge_context",
    "resolve_molecular_pool",
    "run_ap_mclachlan",
    "run_bridge",
    "run_exact_reference",
    "run_mapping_witness",
    "run_qse",
    "run_static_adapt_screening",
    "verified_status",
]
