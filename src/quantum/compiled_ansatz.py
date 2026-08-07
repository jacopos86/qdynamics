"""Compiled ansatz execution helpers (exyz convention).

This module provides a wrapper executor that prepares ansatz states from
``AnsatzTerm``-like inputs using shared compiled Pauli actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
try:  # SciPy is available in production runners; keep import optional for light tooling.
    from scipy import sparse as _scipy_sparse
    from scipy.sparse.linalg import expm_multiply as _scipy_expm_multiply
except Exception:  # pragma: no cover - exercised only in minimal environments.
    _scipy_sparse = None
    _scipy_expm_multiply = None

from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    RotationTermSpec,
    build_parameter_layout,
    iter_runtime_rotation_terms,
)
from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_compiled_pauli,
    apply_compiled_pauli_to_columns,
    apply_exp_term,
    compile_pauli_action_exyz,
    materialize_compiled_pauli_action,
    materialize_compiled_pauli_action_on_inputs,
)

if TYPE_CHECKING:  # pragma: no cover
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm


# The generic ADAPT runner shares one Pauli-action cache for the lifetime of a
# run.  Keep grouped-exact eigensystems in a private nested cache on that same
# object so repeated parent generators and successive refits do not repeat an
# identical dense eigendecomposition.
_GROUPED_EXACT_CACHE_SENTINEL = object()


def _apply_exp_term_to_columns(
    values: np.ndarray,
    action: CompiledPauliAction,
    *,
    coeff: complex,
    dt: float,
    tol: float,
) -> np.ndarray:
    """Apply one Pauli rotation to a matrix of statevector columns."""

    matrix = np.asarray(values, dtype=complex)
    if matrix.ndim != 2:
        raise ValueError(f"values must be a rank-2 column matrix; got {matrix.shape}.")
    coeff_c = complex(coeff)
    if abs(coeff_c.imag) > float(tol):
        raise ValueError(
            f"Imaginary coefficient encountered for {action.label_exyz}: {coeff_c}"
        )
    if int(matrix.shape[0]) != int(action.dimension):
        raise ValueError(
            "Pauli action dimension does not match tangent column matrix: "
            f"{action.dimension} vs {matrix.shape[0]}."
        )
    pauli_values = apply_compiled_pauli_to_columns(matrix, action)
    angle = float(dt) * float(coeff_c.real)
    return np.asarray(
        np.cos(angle) * matrix - 1.0j * np.sin(angle) * pauli_values,
        dtype=complex,
    )


@dataclass(frozen=True)
class CompiledRotationStep:
    coeff_real: float
    action: CompiledPauliAction


@dataclass
class CompiledPolynomialRotationPlan:
    nq: int | None
    label: str
    steps: tuple[CompiledRotationStep, ...]
    runtime_start: int
    execution_mode: str = "termwise_product"
    exact_eigvals: np.ndarray | None = None
    exact_eigvecs: np.ndarray | None = None
    exact_eigvecs_dagger: np.ndarray | None = None
    exact_sparse_matrix: Any | None = None
    grouped_exact_cache_key: tuple[Any, ...] | None = None
    grouped_exact_backend: str | None = None

    @property
    def runtime_count(self) -> int:
        return int(len(self.steps))

    @property
    def runtime_stop(self) -> int:
        return int(self.runtime_start + len(self.steps))


class CompiledAnsatzExecutor:
    """Compiled-action ansatz executor for lists of AnsatzTerm-like objects."""

    GROUPED_EXACT_MAX_DIM = 1024
    GROUPED_EXACT_PLAN_CACHE_ID = "coefficient_ordered_shared_eigensystem_cache_v1"
    PREFIX_STATE_CACHE_ID = "exact_unchanged_theta_prefix_state_reuse_v1"

    _MATH_INIT = (
        r"\\exp(-i\\theta_k H_k)\\approx\\prod_j \\exp(-i\\theta_k c_{k,j} P_{k,j}),"
        r"\\quad P_{k,j}\\mapsto (\\mathrm{perm},\\mathrm{phase})"
    )

    def __init__(
        self,
        terms: Sequence["AnsatzTerm"],
        *,
        coefficient_tolerance: float = 1e-12,
        ignore_identity: bool = True,
        sort_terms: bool = True,
        pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
        parameterization_mode: Literal["logical_shared", "per_pauli_term"] = "logical_shared",
        parameterization_layout: AnsatzParameterLayout | None = None,
        enable_prefix_state_cache: bool = False,
        invariant_basis_indices: np.ndarray | None = None,
    ):
        self.coefficient_tolerance = float(coefficient_tolerance)
        self.ignore_identity = bool(ignore_identity)
        self.sort_terms = bool(sort_terms)
        self.parameterization_mode = str(parameterization_mode)
        self.enable_prefix_state_cache = bool(enable_prefix_state_cache)
        self.invariant_basis_indices = self._normalize_invariant_basis_indices(
            invariant_basis_indices
        )
        self.invariant_basis_fingerprint = (
            None
            if self.invariant_basis_indices is None
            else hashlib.sha256(
                np.asarray(
                    self.invariant_basis_indices,
                    dtype="<i8",
                ).tobytes()
            ).hexdigest()
        )
        self.terms = list(terms)
        self._prefix_cache_theta: np.ndarray | None = None
        self._prefix_cache_states: tuple[np.ndarray, ...] | None = None
        self._prefix_cache_reference_id: int | None = None
        self.prefix_state_cache_evaluation_count = 0
        self.prefix_state_cache_hit_count = 0
        self.prefix_state_cache_reused_operator_count = 0

        self.pauli_action_cache = (
            pauli_action_cache if pauli_action_cache is not None else {}
        )
        grouped_cache = self.pauli_action_cache.get(_GROUPED_EXACT_CACHE_SENTINEL)  # type: ignore[arg-type]
        if not isinstance(grouped_cache, dict):
            grouped_cache = {}
            self.pauli_action_cache[_GROUPED_EXACT_CACHE_SENTINEL] = grouped_cache  # type: ignore[index]
        self.grouped_exact_cache: dict[tuple[Any, ...], tuple[Any, Any, Any, Any]] = grouped_cache
        if parameterization_layout is None:
            self.layout = build_parameter_layout(
                self.terms,
                ignore_identity=self.ignore_identity,
                coefficient_tolerance=self.coefficient_tolerance,
                sort_terms=self.sort_terms,
            )
        else:
            self.layout = parameterization_layout
            self._validate_parameterization_layout(self.layout)
        self._plans: list[CompiledPolynomialRotationPlan] = []
        self.nq: int | None = None
        self.logical_parameter_count = int(self.layout.logical_parameter_count)
        self.runtime_parameter_count = int(self.layout.runtime_parameter_count)
        self.num_parameters = (
            int(self.runtime_parameter_count)
            if self.parameterization_mode == "per_pauli_term"
            else int(self.logical_parameter_count)
        )

        for block, term in zip(self.layout.blocks, self.terms):
            execution_mode = str(
                getattr(term, "execution_mode", "termwise_product") or "termwise_product"
            ).strip().lower()
            if self.parameterization_mode == "per_pauli_term" and execution_mode == "grouped_exact":
                raise ValueError(
                    "grouped_exact execution is incompatible with parameterization_mode='per_pauli_term'; "
                    "use logical_shared or drop/project the generator before per-Pauli execution"
                )
            if parameterization_layout is None:
                plan = self._compile_polynomial_plan(
                    getattr(term, "polynomial"),
                    label=str(block.candidate_label),
                    runtime_start=int(block.runtime_start),
                    execution_mode=execution_mode,
                )
            else:
                plan = self._compile_rotation_specs(
                    tuple(block.terms),
                    poly=getattr(term, "polynomial"),
                    label=str(block.candidate_label),
                    runtime_start=int(block.runtime_start),
                    execution_mode=execution_mode,
                )
            if plan.nq is not None:
                if self.nq is None:
                    self.nq = int(plan.nq)
                elif int(plan.nq) != int(self.nq):
                    raise ValueError(
                        f"Inconsistent ansatz qubit counts: saw {plan.nq} after {self.nq}."
                    )
            self._plans.append(plan)

    @staticmethod
    def _normalize_invariant_basis_indices(
        indices: np.ndarray | None,
    ) -> np.ndarray | None:
        if indices is None:
            return None
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                "invariant_basis_indices must be a nonempty one-dimensional array."
            )
        if values.size > 1 and np.any(values[1:] <= values[:-1]):
            raise ValueError(
                "invariant_basis_indices must be strictly increasing and unique."
            )
        normalized = np.asarray(values, dtype=np.int64).copy()
        normalized.flags.writeable = False
        return normalized

    _MATH_COMPILE_POLYNOMIAL_PLAN = (
        r"H=\\sum_j c_j P_j,\\ \\text{ordered as in runtime termwise execution},"
        r"\\ \\text{skip }|c_j|<\\epsilon,\\ \\text{optional skip }I"
    )

    def _compile_rotation_specs(
        self,
        specs: Sequence[RotationTermSpec],
        *,
        poly: Any,
        label: str,
        runtime_start: int,
        execution_mode: str = "termwise_product",
    ) -> CompiledPolynomialRotationPlan:
        if len(specs) == 0:
            poly_terms = list(poly.return_polynomial())
            nq = int(poly_terms[0].nqubit()) if poly_terms else None
            return CompiledPolynomialRotationPlan(
                nq=nq,
                label=str(label),
                steps=tuple(),
                runtime_start=int(runtime_start),
                execution_mode="termwise_product",
            )

        compiled_steps: list[CompiledRotationStep] = []
        for spec in specs:
            action = self.pauli_action_cache.get(spec.pauli_exyz)
            if action is None:
                action = compile_pauli_action_exyz(spec.pauli_exyz, int(spec.nq))
                self.pauli_action_cache[spec.pauli_exyz] = action
            compiled_steps.append(
                CompiledRotationStep(coeff_real=float(spec.coeff_real), action=action)
            )

        mode = (
            "grouped_exact"
            if str(execution_mode).strip().lower() == "grouped_exact"
            else "termwise_product"
        )
        exact_eigvals: np.ndarray | None = None
        exact_eigvecs: np.ndarray | None = None
        exact_eigvecs_dagger: np.ndarray | None = None
        exact_sparse_matrix: Any | None = None
        grouped_exact_cache_key: tuple[Any, ...] | None = None
        grouped_exact_backend: str | None = None
        if mode == "grouped_exact":
            full_dim = 1 << int(specs[0].nq)
            if self.invariant_basis_indices is not None and (
                int(self.invariant_basis_indices[0]) < 0
                or int(self.invariant_basis_indices[-1]) >= full_dim
            ):
                raise ValueError(
                    "invariant_basis_indices lie outside the ansatz qubit dimension."
                )
            effective_dim = (
                full_dim
                if self.invariant_basis_indices is None
                else int(self.invariant_basis_indices.size)
            )
            grouped_exact_backend = (
                "dense"
                if effective_dim <= int(self.GROUPED_EXACT_MAX_DIM)
                else "sparse"
            )
            grouped_exact_cache_key = (
                self.GROUPED_EXACT_PLAN_CACHE_ID,
                int(specs[0].nq),
                float(self.coefficient_tolerance),
                tuple((str(spec.pauli_exyz), float(spec.coeff_real)) for spec in specs),
                grouped_exact_backend,
                self.invariant_basis_fingerprint,
            )
            cached = self.grouped_exact_cache.get(
                grouped_exact_cache_key
            )
            if cached is not None:
                (
                    exact_eigvals,
                    exact_eigvecs,
                    exact_eigvecs_dagger,
                    exact_sparse_matrix,
                ) = cached

        return CompiledPolynomialRotationPlan(
            nq=int(specs[0].nq),
            label=str(label),
            steps=tuple(compiled_steps),
            runtime_start=int(runtime_start),
            execution_mode=mode,
            exact_eigvals=exact_eigvals,
            exact_eigvecs=exact_eigvecs,
            exact_eigvecs_dagger=exact_eigvecs_dagger,
            exact_sparse_matrix=exact_sparse_matrix,
            grouped_exact_cache_key=grouped_exact_cache_key,
            grouped_exact_backend=grouped_exact_backend,
        )

    def _compile_polynomial_plan(
        self,
        poly: Any,
        *,
        label: str,
        runtime_start: int,
        execution_mode: str = "termwise_product",
    ) -> CompiledPolynomialRotationPlan:
        layout_single = build_parameter_layout(
            [type("_TermCarrier", (), {"label": label, "polynomial": poly})()],
            ignore_identity=self.ignore_identity,
            coefficient_tolerance=self.coefficient_tolerance,
            sort_terms=self.sort_terms,
        )
        block = layout_single.blocks[0] if layout_single.blocks else None
        return self._compile_rotation_specs(
            tuple(block.terms) if block is not None else tuple(),
            poly=poly,
            label=str(label),
            runtime_start=int(runtime_start),
            execution_mode=execution_mode,
        )

    def _compile_grouped_exact_eigendecomposition(
        self,
        steps: Sequence[CompiledRotationStep],
        *,
        nq: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compile H=sum_j c_j P_j into a dense eigendecomposition.

        This path is used only for legal-subspace-hardened grouped generators
        whose Pauli components would leak under finite termwise-product
        execution.  It implements the mathematical grouped unitary exactly for
        the local statevector simulator: exp(-i theta H).
        """

        full_dim = 1 << int(nq)
        basis = self.invariant_basis_indices
        dim = full_dim if basis is None else int(basis.size)
        if dim > int(self.GROUPED_EXACT_MAX_DIM):
            raise ValueError(
                "grouped_exact dense compilation exceeds safety limit: "
                f"dim={dim} > {int(self.GROUPED_EXACT_MAX_DIM)} for nq={int(nq)}"
            )
        cols = np.arange(dim, dtype=np.int64)
        hmat = np.zeros((dim, dim), dtype=complex)
        if basis is None:
            for step in steps:
                permutation, phase = materialize_compiled_pauli_action(
                    step.action
                )
                hmat[permutation, cols] += (
                    float(step.coeff_real) * phase
                )
        else:
            lookup = np.full(full_dim, -1, dtype=np.int32)
            lookup[basis] = np.arange(dim, dtype=np.int32)
            for step in steps:
                output_full, phase = (
                    materialize_compiled_pauli_action_on_inputs(
                        step.action,
                        basis,
                    )
                )
                rows = lookup[output_full]
                valid = rows >= 0
                hmat[rows[valid], cols[valid]] += (
                    float(step.coeff_real) * phase[valid]
                )
        hermitian_error = float(np.max(np.abs(hmat - hmat.conj().T))) if dim else 0.0
        hermitian_tol = max(1e-10, 100.0 * float(self.coefficient_tolerance))
        if hermitian_error > hermitian_tol:
            raise ValueError(
                f"grouped_exact generator is not Hermitian within tolerance: {hermitian_error:.3e}"
            )
        hmat = 0.5 * (hmat + hmat.conj().T)
        eigvals, eigvecs = np.linalg.eigh(hmat)
        return np.asarray(eigvals, dtype=float), np.asarray(eigvecs, dtype=complex)

    def _compile_grouped_exact_sparse_matrix(
        self,
        steps: Sequence[CompiledRotationStep],
        *,
        nq: int,
    ) -> Any:
        """Compile H=sum_j c_j P_j into a sparse Hermitian matrix.

        Large molecular-vibronic grouped generators cannot use the dense
        eigendecomposition path.  The sparse matrix is applied later through
        Krylov expm_multiply, preserving the grouped exponential without
        allocating a dense 2^n by 2^n matrix.
        """

        if _scipy_sparse is None or _scipy_expm_multiply is None:
            raise ValueError(
                "grouped_exact sparse execution requires scipy.sparse.linalg.expm_multiply "
                f"for nq={int(nq)}"
            )
        full_dim = 1 << int(nq)
        basis = self.invariant_basis_indices
        dim = full_dim if basis is None else int(basis.size)
        if not steps:
            return _scipy_sparse.csr_matrix((dim, dim), dtype=complex)
        cols = np.arange(dim, dtype=np.int64)
        input_indices = (
            np.arange(full_dim, dtype=np.int64)
            if basis is None
            else basis
        )
        lookup = None
        if basis is not None:
            lookup = np.full(full_dim, -1, dtype=np.int32)
            lookup[basis] = np.arange(dim, dtype=np.int32)

        steps_by_flip: dict[int, list[CompiledRotationStep]] = {}
        for step in steps:
            steps_by_flip.setdefault(
                int(step.action.flip_mask),
                [],
            ).append(step)

        hmat = _scipy_sparse.csr_matrix((dim, dim), dtype=complex)
        rows: list[np.ndarray] = []
        col_blocks: list[np.ndarray] = []
        data: list[np.ndarray] = []
        buffered_entries = 0
        maximum_buffered_entries = 2_000_000

        def flush_blocks() -> None:
            nonlocal hmat, buffered_entries
            if not rows:
                return
            chunk = _scipy_sparse.coo_matrix(
                (
                    np.concatenate(data),
                    (
                        np.concatenate(rows),
                        np.concatenate(col_blocks),
                    ),
                ),
                shape=(dim, dim),
                dtype=complex,
            ).tocsr()
            chunk.sum_duplicates()
            chunk.eliminate_zeros()
            hmat = (hmat + chunk).tocsr()
            rows.clear()
            col_blocks.clear()
            data.clear()
            buffered_entries = 0

        for flip_mask, grouped_steps in steps_by_flip.items():
            output_full = np.bitwise_xor(
                input_indices,
                np.int64(flip_mask),
            )
            amplitudes = np.zeros(dim, dtype=complex)
            for step in grouped_steps:
                _, phase = materialize_compiled_pauli_action_on_inputs(
                    step.action,
                    input_indices,
                )
                amplitudes += float(step.coeff_real) * phase
            if basis is None:
                block_rows = output_full
                block_cols = cols
                block_data = amplitudes
            else:
                reduced_rows = lookup[output_full]
                valid = reduced_rows >= 0
                block_rows = np.asarray(
                    reduced_rows[valid],
                    dtype=np.int64,
                )
                block_cols = cols[valid]
                block_data = amplitudes[valid]
            rows.append(np.asarray(block_rows, dtype=np.int64))
            col_blocks.append(np.asarray(block_cols, dtype=np.int64))
            data.append(np.asarray(block_data, dtype=complex))
            buffered_entries += int(block_data.size)
            if buffered_entries >= maximum_buffered_entries:
                flush_blocks()
        flush_blocks()
        hmat.sum_duplicates()
        hmat.eliminate_zeros()
        hermitian_delta = (hmat - hmat.getH()).tocoo()
        hermitian_error = (
            float(np.max(np.abs(hermitian_delta.data)))
            if hermitian_delta.nnz
            else 0.0
        )
        hermitian_tol = max(1e-10, 100.0 * float(self.coefficient_tolerance))
        if hermitian_error > hermitian_tol:
            raise ValueError(
                f"grouped_exact sparse generator is not Hermitian within tolerance: {hermitian_error:.3e}"
            )
        return (0.5 * (hmat + hmat.getH())).tocsr()

    def _apply_grouped_exact(
        self,
        psi: np.ndarray,
        plan: CompiledPolynomialRotationPlan,
        dt: float,
    ) -> np.ndarray:
        self._ensure_grouped_exact_plan(plan)
        state = np.asarray(psi, dtype=complex).reshape(-1)
        basis = self.invariant_basis_indices
        if basis is None:
            working_state = state
        else:
            sector_state = np.asarray(state[basis], dtype=complex)
            outside_norm_sq = max(
                0.0,
                float(np.vdot(state, state).real)
                - float(np.vdot(sector_state, sector_state).real),
            )
            if outside_norm_sq > 1.0e-12:
                raise ValueError(
                    "Grouped exact invariant-basis execution received "
                    f"out-of-sector norm squared {outside_norm_sq:.3e}."
                )
            working_state = sector_state
        if plan.exact_eigvals is None or plan.exact_eigvecs is None:
            if plan.exact_sparse_matrix is None:
                return state
            if _scipy_expm_multiply is None:
                raise ValueError("grouped_exact sparse execution requires scipy.sparse.linalg.expm_multiply")
            evolved = np.asarray(
                _scipy_expm_multiply(
                    (-1.0j * float(dt)) * plan.exact_sparse_matrix,
                    working_state,
                ),
                dtype=complex,
            )
        else:
            eigvecs_dagger = plan.exact_eigvecs_dagger
            if eigvecs_dagger is None:
                eigvecs_dagger = plan.exact_eigvecs.conj().T
            coeffs = np.asarray(
                eigvecs_dagger @ working_state,
                dtype=complex,
            )
            coeffs = (
                np.exp(-1.0j * float(dt) * plan.exact_eigvals)
                * coeffs
            )
            evolved = np.asarray(
                plan.exact_eigvecs @ coeffs,
                dtype=complex,
            )
        if basis is None:
            return evolved
        out = np.zeros_like(state, dtype=complex)
        out[basis] = evolved
        return out

    def _ensure_grouped_exact_plan(
        self,
        plan: CompiledPolynomialRotationPlan,
    ) -> None:
        if (
            plan.exact_eigvals is not None
            or plan.exact_sparse_matrix is not None
        ):
            return
        cache_key = plan.grouped_exact_cache_key
        backend = plan.grouped_exact_backend
        if cache_key is None or backend not in {"dense", "sparse"}:
            raise RuntimeError(
                "Grouped exact plan omitted its lazy compilation identity."
            )
        cached = self.grouped_exact_cache.get(cache_key)
        if cached is None:
            exact_eigvals: np.ndarray | None = None
            exact_eigvecs: np.ndarray | None = None
            exact_eigvecs_dagger: np.ndarray | None = None
            exact_sparse_matrix: Any | None = None
            if backend == "dense":
                exact_eigvals, exact_eigvecs = (
                    self._compile_grouped_exact_eigendecomposition(
                        plan.steps,
                        nq=int(plan.nq),
                    )
                )
                exact_eigvecs_dagger = exact_eigvecs.conj().T
            else:
                exact_sparse_matrix = (
                    self._compile_grouped_exact_sparse_matrix(
                        plan.steps,
                        nq=int(plan.nq),
                    )
                )
            cached = (
                exact_eigvals,
                exact_eigvecs,
                exact_eigvecs_dagger,
                exact_sparse_matrix,
            )
            self.grouped_exact_cache[cache_key] = cached
        (
            plan.exact_eigvals,
            plan.exact_eigvecs,
            plan.exact_eigvecs_dagger,
            plan.exact_sparse_matrix,
        ) = cached

    def _apply_plan(
        self,
        psi: np.ndarray,
        plan: CompiledPolynomialRotationPlan,
        dt: float,
    ) -> np.ndarray:
        if float(dt) == 0.0:
            return np.asarray(psi, dtype=complex)
        if str(plan.execution_mode).strip().lower() == "grouped_exact":
            return self._apply_grouped_exact(np.asarray(psi, dtype=complex), plan, float(dt))
        out = np.asarray(psi, dtype=complex)
        for step in plan.steps:
            out = apply_exp_term(
                out,
                step.action,
                coeff=complex(step.coeff_real),
                dt=float(dt),
                tol=self.coefficient_tolerance,
            )
        return np.asarray(out, dtype=complex)

    @staticmethod
    def _apply_generator_hamiltonian(
        psi: np.ndarray,
        plan: CompiledPolynomialRotationPlan,
    ) -> np.ndarray:
        out = np.zeros_like(np.asarray(psi, dtype=complex))
        for step in plan.steps:
            out = out + float(step.coeff_real) * apply_compiled_pauli(psi, step.action)
        return np.asarray(out, dtype=complex)

    def _apply_plan_with_logical_tangent(
        self,
        psi: np.ndarray,
        plan: CompiledPolynomialRotationPlan,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        psi_in = np.asarray(psi, dtype=complex)
        if str(plan.execution_mode).strip().lower() == "grouped_exact":
            psi_out = self._apply_plan(psi_in, plan, float(dt))
            tangent = -1.0j * self._apply_generator_hamiltonian(psi_out, plan)
            return np.asarray(psi_out, dtype=complex), np.asarray(tangent, dtype=complex)

        prefix_after: list[np.ndarray] = []
        psi_work = psi_in.copy()
        for step in plan.steps:
            if float(dt) != 0.0:
                psi_work = apply_exp_term(
                    psi_work,
                    step.action,
                    coeff=complex(step.coeff_real),
                    dt=float(dt),
                    tol=self.coefficient_tolerance,
                )
            prefix_after.append(np.asarray(psi_work, dtype=complex).copy())

        tangent_total = np.zeros_like(psi_in)
        for local_idx, step in enumerate(plan.steps):
            tangent = (
                -1.0j
                * float(step.coeff_real)
                * apply_compiled_pauli(prefix_after[local_idx], step.action)
            )
            for suffix_step in plan.steps[int(local_idx) + 1 :]:
                if float(dt) == 0.0:
                    continue
                tangent = apply_exp_term(
                    tangent,
                    suffix_step.action,
                    coeff=complex(suffix_step.coeff_real),
                    dt=float(dt),
                    tol=self.coefficient_tolerance,
                )
            tangent_total = tangent_total + tangent
        return np.asarray(psi_work, dtype=complex), np.asarray(tangent_total, dtype=complex)

    def _validate_parameterization_layout(
        self,
        layout: AnsatzParameterLayout,
    ) -> None:
        if int(layout.logical_parameter_count) != int(len(self.terms)):
            raise ValueError(
                f"parameterization logical count mismatch: {layout.logical_parameter_count} vs {len(self.terms)} terms."
            )
        runtime_start_expected = 0
        coeff_tol = max(1e-12, 10.0 * float(self.coefficient_tolerance))
        use_sorted_order = str(layout.term_order).strip().lower() == "sorted"
        for logical_index, (block, term) in enumerate(zip(layout.blocks, self.terms)):
            if int(block.logical_index) != int(logical_index):
                raise ValueError(
                    f"parameterization block logical_index mismatch: got {block.logical_index}, expected {logical_index}."
                )
            if int(block.runtime_start) != int(runtime_start_expected):
                raise ValueError(
                    f"parameterization block runtime_start mismatch: got {block.runtime_start}, expected {runtime_start_expected}."
                )
            term_label = str(getattr(term, "label", f"term_{logical_index}"))
            if str(block.candidate_label) != term_label:
                raise ValueError(
                    f"parameterization block label mismatch at {logical_index}: {block.candidate_label!r} vs {term_label!r}."
                )
            expected_specs = iter_runtime_rotation_terms(
                getattr(term, "polynomial"),
                ignore_identity=bool(layout.ignore_identity),
                coefficient_tolerance=float(layout.coefficient_tolerance),
                sort_terms=use_sorted_order,
            )
            if len(expected_specs) != int(block.runtime_count):
                raise ValueError(
                    f"parameterization block runtime_count mismatch for {term_label}: {block.runtime_count} vs expected {len(expected_specs)}."
                )
            for spec_expected, spec_actual in zip(expected_specs, block.terms):
                if str(spec_expected.pauli_exyz) != str(spec_actual.pauli_exyz):
                    raise ValueError(
                        f"parameterization runtime term label mismatch for {term_label}: {spec_actual.pauli_exyz!r} vs expected {spec_expected.pauli_exyz!r}."
                    )
                if int(spec_expected.nq) != int(spec_actual.nq):
                    raise ValueError(
                        f"parameterization runtime term nq mismatch for {term_label}: {spec_actual.nq} vs expected {spec_expected.nq}."
                    )
                if abs(float(spec_expected.coeff_real) - float(spec_actual.coeff_real)) > coeff_tol:
                    raise ValueError(
                        f"parameterization runtime term coefficient mismatch for {term_label}: {spec_actual.coeff_real} vs expected {spec_expected.coeff_real}."
                    )
            runtime_start_expected += int(block.runtime_count)

    _MATH_PREPARE_STATE = (
        r"|\\psi(\\vartheta)\\rangle=\\prod_k \\prod_j \\exp(-i\\vartheta_{k,j} c_{k,j}P_{k,j})|\\psi_{\\mathrm{ref}}\\rangle"
    )

    def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        """Prepare ansatz state using compiled Pauli actions."""
        if self.enable_prefix_state_cache:
            self.prefix_state_cache_evaluation_count += 1
        theta_vec = np.asarray(theta, dtype=float).reshape(-1)
        expected = (
            int(self.runtime_parameter_count)
            if self.parameterization_mode == "per_pauli_term"
            else int(self.logical_parameter_count)
        )
        if int(theta_vec.size) != int(expected):
            raise ValueError(
                f"theta length mismatch: got {theta_vec.size}, expected {expected}."
            )

        psi_ref_array = np.asarray(psi_ref, dtype=complex)
        psi_ref_id = id(psi_ref)
        psi = psi_ref_array.reshape(-1).copy()
        if self.nq is not None:
            expected_dim = 1 << int(self.nq)
            if int(psi.size) != int(expected_dim):
                raise ValueError(
                    f"psi_ref length mismatch: got {psi.size}, expected {expected_dim} for nq={self.nq}."
                )

        if self.parameterization_mode == "per_pauli_term":
            for poly_plan in self._plans:
                if int(poly_plan.runtime_count) <= 0:
                    continue
                block_theta = theta_vec[poly_plan.runtime_start:poly_plan.runtime_stop]
                for local_idx, step in enumerate(poly_plan.steps):
                    dt = float(block_theta[local_idx])
                    if dt == 0.0:
                        continue
                    psi = apply_exp_term(
                        psi,
                        step.action,
                        coeff=complex(step.coeff_real),
                        dt=dt,
                        tol=self.coefficient_tolerance,
                    )
            return psi

        start_index = 0
        prefix_states: list[np.ndarray] = [psi.copy()]
        if (
            self.enable_prefix_state_cache
            and self._prefix_cache_theta is not None
            and self._prefix_cache_states is not None
            and self._prefix_cache_reference_id == psi_ref_id
            and self._prefix_cache_theta.shape == theta_vec.shape
            and len(self._prefix_cache_states) == len(self._plans) + 1
        ):
            changed = np.flatnonzero(theta_vec != self._prefix_cache_theta)
            start_index = int(changed[0]) if changed.size else len(self._plans)
            self.prefix_state_cache_hit_count += 1
            self.prefix_state_cache_reused_operator_count += int(start_index)
            prefix_states = [state for state in self._prefix_cache_states[: start_index + 1]]
            psi = np.asarray(prefix_states[-1], dtype=complex).copy()

        for k, poly_plan in enumerate(self._plans[start_index:], start=start_index):
            dt = float(theta_vec[k])
            psi = self._apply_plan(psi, poly_plan, dt)
            prefix_states.append(np.asarray(psi, dtype=complex).copy())
        if self.enable_prefix_state_cache:
            self._prefix_cache_theta = theta_vec.copy()
            self._prefix_cache_states = tuple(prefix_states)
            self._prefix_cache_reference_id = psi_ref_id
        return psi

    _MATH_PREPARE_STATE_WITH_RUNTIME_TANGENTS = (
        r"\\partial_{\\theta_j}|\\psi(\\theta)\\rangle="
        r"\\Big(\\prod_{k>j}U_k(\\theta_k)\\Big)(-i c_j P_j)\\Big(\\prod_{k\\le j}U_k(\\theta_k)\\Big)|\\psi_{\\mathrm{ref}}\\rangle"
    )

    def prepare_state_with_runtime_tangents(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        runtime_indices: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        if self.parameterization_mode != "per_pauli_term":
            raise ValueError(
                "prepare_state_with_runtime_tangents requires parameterization_mode='per_pauli_term'."
            )
        theta_vec = np.asarray(theta, dtype=float).reshape(-1)
        if int(theta_vec.size) != int(self.runtime_parameter_count):
            raise ValueError(
                f"theta length mismatch: got {theta_vec.size}, expected {self.runtime_parameter_count}."
            )

        requested = (
            list(range(int(self.runtime_parameter_count)))
            if runtime_indices is None
            else [int(idx) for idx in runtime_indices]
        )
        requested_unique = sorted(set(requested))
        for idx in requested_unique:
            if idx < 0 or idx >= int(self.runtime_parameter_count):
                raise ValueError(
                    f"runtime index {idx} out of bounds for runtime_parameter_count={self.runtime_parameter_count}."
                )

        if requested_unique == list(range(int(self.runtime_parameter_count))):
            return self._prepare_state_with_all_runtime_tangents_batched(
                theta_vec,
                psi_ref,
            )

        psi = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
        if self.nq is not None:
            expected_dim = 1 << int(self.nq)
            if int(psi.size) != int(expected_dim):
                raise ValueError(
                    f"psi_ref length mismatch: got {psi.size}, expected {expected_dim} for nq={self.nq}."
                )

        steps_by_runtime: dict[int, CompiledRotationStep] = {}
        prefix_after: dict[int, np.ndarray] = {}
        requested_set = set(requested_unique)
        for poly_plan in self._plans:
            for local_idx, step in enumerate(poly_plan.steps):
                runtime_idx = int(poly_plan.runtime_start + local_idx)
                steps_by_runtime[runtime_idx] = step
                dt = float(theta_vec[runtime_idx])
                if dt != 0.0:
                    psi = apply_exp_term(
                        psi,
                        step.action,
                        coeff=complex(step.coeff_real),
                        dt=dt,
                        tol=self.coefficient_tolerance,
                    )
                if runtime_idx in requested_set:
                    prefix_after[runtime_idx] = np.asarray(psi, dtype=complex).copy()

        tangents: dict[int, np.ndarray] = {}
        for runtime_idx in requested_unique:
            step = steps_by_runtime[runtime_idx]
            tangent = (
                -1.0j
                * float(step.coeff_real)
                * apply_compiled_pauli(prefix_after[runtime_idx], step.action)
            )
            for suffix_idx in range(int(runtime_idx) + 1, int(self.runtime_parameter_count)):
                suffix_step = steps_by_runtime[suffix_idx]
                dt = float(theta_vec[suffix_idx])
                if dt == 0.0:
                    continue
                tangent = apply_exp_term(
                    tangent,
                    suffix_step.action,
                    coeff=complex(suffix_step.coeff_real),
                    dt=dt,
                    tol=self.coefficient_tolerance,
                )
            tangents[int(runtime_idx)] = np.asarray(tangent, dtype=complex)

        return np.asarray(psi, dtype=complex), tangents

    def _prepare_state_with_all_runtime_tangents_batched(
        self,
        theta_vec: np.ndarray,
        psi_ref: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Prepare all runtime tangents with batched suffix propagation."""

        psi = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
        if self.nq is not None:
            expected_dim = 1 << int(self.nq)
            if int(psi.size) != int(expected_dim):
                raise ValueError(
                    f"psi_ref length mismatch: got {psi.size}, expected {expected_dim} for nq={self.nq}."
                )

        runtime_count = int(self.runtime_parameter_count)
        tangent_matrix = np.zeros((int(psi.size), runtime_count), dtype=complex)
        steps_by_runtime: list[CompiledRotationStep | None] = [None] * runtime_count
        for poly_plan in self._plans:
            for local_idx, step in enumerate(poly_plan.steps):
                runtime_idx = int(poly_plan.runtime_start + local_idx)
                steps_by_runtime[runtime_idx] = step
                dt = float(theta_vec[runtime_idx])
                if dt != 0.0:
                    psi = apply_exp_term(
                        psi,
                        step.action,
                        coeff=complex(step.coeff_real),
                        dt=dt,
                        tol=self.coefficient_tolerance,
                    )
                tangent_matrix[:, runtime_idx] = (
                    -1.0j
                    * float(step.coeff_real)
                    * apply_compiled_pauli(psi, step.action)
                )

        for runtime_idx, step in enumerate(steps_by_runtime):
            if step is None:
                raise ValueError(
                    f"missing compiled rotation step for runtime index {runtime_idx}."
                )
            if runtime_idx == 0:
                continue
            dt = float(theta_vec[runtime_idx])
            if dt == 0.0:
                continue
            tangent_matrix[:, :runtime_idx] = _apply_exp_term_to_columns(
                tangent_matrix[:, :runtime_idx],
                step.action,
                coeff=complex(step.coeff_real),
                dt=dt,
                tol=self.coefficient_tolerance,
            )

        tangents = {
            int(runtime_idx): np.asarray(
                tangent_matrix[:, runtime_idx],
                dtype=complex,
            ).copy()
            for runtime_idx in range(runtime_count)
        }
        return np.asarray(psi, dtype=complex), tangents

    def prepare_state_with_parameter_tangents(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        parameter_indices: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Prepare state and tangent columns for the active parameterization mode."""

        if self.parameterization_mode == "per_pauli_term":
            return self.prepare_state_with_runtime_tangents(
                theta,
                psi_ref,
                runtime_indices=parameter_indices,
            )
        if self.parameterization_mode != "logical_shared":
            raise ValueError(f"Unsupported parameterization_mode: {self.parameterization_mode!r}")

        theta_vec = np.asarray(theta, dtype=float).reshape(-1)
        if int(theta_vec.size) != int(self.logical_parameter_count):
            raise ValueError(
                f"theta length mismatch: got {theta_vec.size}, expected {self.logical_parameter_count}."
            )
        requested = (
            list(range(int(self.logical_parameter_count)))
            if parameter_indices is None
            else [int(idx) for idx in parameter_indices]
        )
        requested_unique = sorted(set(requested))
        for idx in requested_unique:
            if idx < 0 or idx >= int(self.logical_parameter_count):
                raise ValueError(
                    f"parameter index {idx} out of bounds for logical_parameter_count={self.logical_parameter_count}."
                )

        psi = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
        if self.nq is not None:
            expected_dim = 1 << int(self.nq)
            if int(psi.size) != int(expected_dim):
                raise ValueError(
                    f"psi_ref length mismatch: got {psi.size}, expected {expected_dim} for nq={self.nq}."
                )

        requested_set = set(requested_unique)
        local_tangent_after_plan: dict[int, np.ndarray] = {}
        for logical_idx, plan in enumerate(self._plans):
            dt = float(theta_vec[logical_idx])
            if int(logical_idx) in requested_set:
                psi, tangent = self._apply_plan_with_logical_tangent(psi, plan, dt)
                local_tangent_after_plan[int(logical_idx)] = np.asarray(tangent, dtype=complex)
            else:
                psi = self._apply_plan(psi, plan, dt)

        tangents: dict[int, np.ndarray] = {}
        for logical_idx in requested_unique:
            tangent = np.asarray(local_tangent_after_plan[int(logical_idx)], dtype=complex)
            for suffix_idx in range(int(logical_idx) + 1, int(self.logical_parameter_count)):
                tangent = self._apply_plan(
                    tangent,
                    self._plans[int(suffix_idx)],
                    float(theta_vec[int(suffix_idx)]),
                )
            tangents[int(logical_idx)] = np.asarray(tangent, dtype=complex)

        return np.asarray(psi, dtype=complex), tangents


__all__ = ["CompiledAnsatzExecutor"]
