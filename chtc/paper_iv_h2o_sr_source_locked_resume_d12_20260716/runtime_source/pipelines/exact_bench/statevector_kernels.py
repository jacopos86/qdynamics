"""Shared statevector kernel wrappers for exact_bench pipelines.

These wrappers centralize optimized statevector primitives used by
noise/exact runners and keep a stable interface for report pipelines.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    adapt_commutator_grad_from_hpsi,
    compile_polynomial_action,
    energy_via_one_apply,
)
from src.quantum.pauli_actions import CompiledPauliAction
from src.quantum.vqe_latex_python_pairs import expval_pauli_polynomial_one_apply


def _adapt_apply_h_poly(
    psi: np.ndarray,
    h_poly: Any,
    *,
    compiled: CompiledPolynomialAction | None = None,
) -> np.ndarray:
    # Lazy import avoids eager heavy pipeline module initialization.
    from pipelines.hardcoded import adapt_pipeline as adapt_mod

    return np.asarray(
        adapt_mod._apply_pauli_polynomial(
            np.asarray(psi, dtype=complex).reshape(-1),
            h_poly,
            compiled=compiled,
        ),
        dtype=complex,
    ).reshape(-1)


def _adapt_prepare_state(
    psi_ref: np.ndarray,
    ansatz_ops: Sequence[Any],
    theta: np.ndarray,
) -> np.ndarray:
    # Lazy import avoids eager heavy pipeline module initialization.
    from pipelines.hardcoded import adapt_pipeline as adapt_mod

    return np.asarray(
        adapt_mod._prepare_adapt_state(
            np.asarray(psi_ref, dtype=complex).reshape(-1),
            list(ansatz_ops),
            np.asarray(theta, dtype=float).reshape(-1),
        ),
        dtype=complex,
    ).reshape(-1)


_MATH_APPLY_H_POLY = r"H|\psi\rangle=\sum_j c_j P_j|\psi\rangle"


def apply_h_poly(
    psi: np.ndarray,
    h_poly: Any,
    compiled: CompiledPolynomialAction | None = None,
) -> np.ndarray:
    """Return H|psi> using shared optimized ADAPT kernels."""
    return _adapt_apply_h_poly(psi, h_poly, compiled=compiled)


_MATH_ENERGY_ONE_APPLY = r"E=\operatorname{Re}\langle\psi|H|\psi\rangle"


def energy_one_apply(
    psi: np.ndarray,
    h_poly: Any,
    compiled: CompiledPolynomialAction | None = None,
) -> float:
    """Return one-apply energy using shared compiled/vqe helper paths."""
    psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
    if compiled is not None:
        energy, _hpsi = energy_via_one_apply(psi_vec, compiled)
        return float(energy)
    return float(expval_pauli_polynomial_one_apply(psi_vec, h_poly, tol=1e-12, cache={}))


_MATH_PREPARE_STATE = (
    r"|\psi(\theta)\rangle=\prod_k \exp(-i\theta_k G_k)|\psi_{\mathrm{ref}}\rangle"
)


def prepare_state_for_ansatz(
    psi_ref: np.ndarray,
    ansatz_ops: Sequence[Any],
    theta: np.ndarray,
    *,
    compiled_cache: Any | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Prepare ansatz state using compiled executor when provided/built."""
    psi_ref_vec = np.asarray(psi_ref, dtype=complex).reshape(-1)
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    ops_list = list(ansatz_ops)

    executor: CompiledAnsatzExecutor | None = None
    if isinstance(compiled_cache, CompiledAnsatzExecutor):
        executor = compiled_cache
    elif isinstance(compiled_cache, dict):
        maybe_exec = compiled_cache.get("executor", None)
        if isinstance(maybe_exec, CompiledAnsatzExecutor):
            executor = maybe_exec
        elif ops_list:
            pauli_action_cache = compiled_cache.setdefault("pauli_action_cache", {})
            executor = CompiledAnsatzExecutor(
                ops_list,
                coefficient_tolerance=1e-12,
                ignore_identity=True,
                sort_terms=True,
                pauli_action_cache=pauli_action_cache,
            )
            compiled_cache["executor"] = executor

    if executor is not None:
        psi = np.asarray(executor.prepare_state(theta_vec, psi_ref_vec), dtype=complex).reshape(-1)
    else:
        psi = _adapt_prepare_state(psi_ref_vec, ops_list, theta_vec)

    if not bool(normalize):
        return psi
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero norm state in prepare_state_for_ansatz.")
    return psi / nrm


_MATH_COMMUTATOR_GRAD = r"g=2\,\operatorname{Im}\langle H\psi|A\psi\rangle"


def commutator_gradient(
    pool_op: Any,
    psi: np.ndarray,
    h_poly: Any,
    *,
    compiled: Any | None = None,
    h_psi_precomputed: np.ndarray | None = None,
) -> float:
    """Compute ADAPT commutator gradient with optional compiled artifacts."""
    h_compiled: CompiledPolynomialAction | None = None
    pool_compiled: CompiledPolynomialAction | None = None
    if isinstance(compiled, dict):
        h_compiled = compiled.get("h_compiled", None)
        pool_compiled = compiled.get("pool_compiled", None)
    elif isinstance(compiled, tuple) and len(compiled) == 2:
        h_compiled = compiled[0]
        pool_compiled = compiled[1]

    if h_psi_precomputed is not None:
        a_psi = apply_h_poly(psi, getattr(pool_op, "polynomial"), compiled=pool_compiled)
        return float(adapt_commutator_grad_from_hpsi(np.asarray(h_psi_precomputed, dtype=complex), a_psi))

    # Lazy import avoids eager heavy pipeline module initialization.
    from pipelines.hardcoded import adapt_pipeline as adapt_mod

    return float(
        adapt_mod._commutator_gradient(
            h_poly,
            pool_op,
            np.asarray(psi, dtype=complex).reshape(-1),
            h_compiled=h_compiled,
            pool_compiled=pool_compiled,
            hpsi_precomputed=None,
        )
    )


_MATH_COMPILE_H_POLY = r"C(H)=\mathrm{compile\_polynomial\_action}(H)"


def compile_h_poly(
    h_poly: Any,
    *,
    tol: float = 1e-12,
    pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
) -> CompiledPolynomialAction | None:
    """Compile a PauliPolynomial action when non-empty."""
    terms = h_poly.return_polynomial()
    if not terms:
        return None
    return compile_polynomial_action(
        h_poly,
        tol=float(tol),
        pauli_action_cache=pauli_action_cache,
    )


_MATH_COMPILE_ANSATZ_EXEC = r"\{G_k\}\mapsto \mathrm{CompiledAnsatzExecutor}"


def compile_ansatz_executor(
    ansatz_ops: Sequence[Any],
    *,
    coefficient_tolerance: float = 1e-12,
    ignore_identity: bool = True,
    sort_terms: bool = True,
    pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
) -> CompiledAnsatzExecutor | None:
    """Build a compiled ansatz executor for the given operator list."""
    if len(ansatz_ops) == 0:
        return None
    return CompiledAnsatzExecutor(
        list(ansatz_ops),
        coefficient_tolerance=float(coefficient_tolerance),
        ignore_identity=bool(ignore_identity),
        sort_terms=bool(sort_terms),
        pauli_action_cache=pauli_action_cache,
    )


# Backward-compatible aliases used by earlier callers.
def prepare_state(
    psi_ref: np.ndarray,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    *,
    compiled_ansatz: CompiledAnsatzExecutor | None = None,
) -> np.ndarray:
    return prepare_state_for_ansatz(
        psi_ref,
        selected_ops,
        theta,
        compiled_cache=compiled_ansatz,
        normalize=False,
    )


def commutator_gradient_from_hpsi(hpsi: np.ndarray, apsi: np.ndarray) -> float:
    return float(adapt_commutator_grad_from_hpsi(hpsi, apsi))


__all__ = [
    "apply_h_poly",
    "energy_one_apply",
    "prepare_state_for_ansatz",
    "commutator_gradient",
    "compile_h_poly",
    "compile_ansatz_executor",
    "prepare_state",
    "commutator_gradient_from_hpsi",
]
