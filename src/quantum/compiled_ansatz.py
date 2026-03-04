"""Compiled ansatz execution helpers (exyz convention).

This module provides a wrapper executor that prepares ansatz states from
``AnsatzTerm``-like inputs using shared compiled Pauli actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_exp_term,
    compile_pauli_action_exyz,
)

if TYPE_CHECKING:  # pragma: no cover
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm


@dataclass(frozen=True)
class CompiledRotationStep:
    coeff_real: float
    action: CompiledPauliAction


@dataclass(frozen=True)
class CompiledPolynomialRotationPlan:
    nq: int | None
    steps: tuple[CompiledRotationStep, ...]


class CompiledAnsatzExecutor:
    """Compiled-action ansatz executor for lists of AnsatzTerm-like objects."""

    _MATH_INIT = (
        r"\exp(-i\theta_k H_k)\approx\prod_j \exp(-i\theta_k c_{k,j} P_{k,j}),"
        r"\quad P_{k,j}\mapsto (\mathrm{perm},\mathrm{phase})"
    )

    def __init__(
        self,
        terms: Sequence["AnsatzTerm"],
        *,
        coefficient_tolerance: float = 1e-12,
        ignore_identity: bool = True,
        sort_terms: bool = True,
        pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
    ):
        self.coefficient_tolerance = float(coefficient_tolerance)
        self.ignore_identity = bool(ignore_identity)
        self.sort_terms = bool(sort_terms)
        self.terms = list(terms)

        self.pauli_action_cache = (
            pauli_action_cache if pauli_action_cache is not None else {}
        )
        self._plans: list[CompiledPolynomialRotationPlan] = []
        self.nq: int | None = None

        for term in self.terms:
            plan = self._compile_polynomial_plan(getattr(term, "polynomial"))
            if plan.nq is not None:
                if self.nq is None:
                    self.nq = int(plan.nq)
                elif int(plan.nq) != int(self.nq):
                    raise ValueError(
                        f"Inconsistent ansatz qubit counts: saw {plan.nq} after {self.nq}."
                    )
            self._plans.append(plan)

    _MATH_COMPILE_POLYNOMIAL_PLAN = (
        r"H=\sum_j c_j P_j,\ \text{ordered as in apply\_exp\_pauli\_polynomial},"
        r"\ \text{skip }|c_j|<\epsilon,\ \text{optional skip }I"
    )

    def _compile_polynomial_plan(self, poly: Any) -> CompiledPolynomialRotationPlan:
        poly_terms = list(poly.return_polynomial())
        if not poly_terms:
            return CompiledPolynomialRotationPlan(nq=None, steps=tuple())

        nq = int(poly_terms[0].nqubit())
        id_label = "e" * nq
        ordered = list(poly_terms)
        if self.sort_terms:
            ordered.sort(key=lambda t: t.pw2strng())

        compiled_steps: list[CompiledRotationStep] = []
        for pauli_term in ordered:
            label = str(pauli_term.pw2strng())
            coeff = complex(pauli_term.p_coeff)

            if int(pauli_term.nqubit()) != nq:
                raise ValueError(
                    f"Inconsistent polynomial term qubit count: expected {nq}, got {pauli_term.nqubit()}."
                )
            if len(label) != nq:
                raise ValueError(f"Invalid Pauli label length for '{label}': expected {nq}.")
            if abs(coeff) < self.coefficient_tolerance:
                continue
            if self.ignore_identity and label == id_label:
                continue
            if abs(coeff.imag) > self.coefficient_tolerance:
                raise ValueError(f"Non-negligible imaginary coefficient in term {label}: {coeff}.")

            action = self.pauli_action_cache.get(label)
            if action is None:
                action = compile_pauli_action_exyz(label, nq)
                self.pauli_action_cache[label] = action
            compiled_steps.append(CompiledRotationStep(coeff_real=float(coeff.real), action=action))

        return CompiledPolynomialRotationPlan(nq=nq, steps=tuple(compiled_steps))

    _MATH_PREPARE_STATE = (
        r"|\psi(\theta)\rangle=\prod_k \exp(-i\theta_k H_k)|\psi_{\mathrm{ref}}\rangle"
        r"\approx \prod_k \prod_j \exp(-i\theta_k c_{k,j}P_{k,j})|\psi_{\mathrm{ref}}\rangle"
    )

    def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        """Prepare ansatz state using compiled Pauli actions."""
        theta_vec = np.asarray(theta, dtype=float).reshape(-1)
        if int(theta_vec.size) != int(len(self._plans)):
            raise ValueError(
                f"theta length mismatch: got {theta_vec.size}, expected {len(self._plans)}."
            )

        psi = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
        if self.nq is not None:
            expected_dim = 1 << int(self.nq)
            if int(psi.size) != int(expected_dim):
                raise ValueError(
                    f"psi_ref length mismatch: got {psi.size}, expected {expected_dim} for nq={self.nq}."
                )

        for k, poly_plan in enumerate(self._plans):
            dt = float(theta_vec[k])
            if dt == 0.0:
                continue
            for step in poly_plan.steps:
                psi = apply_exp_term(
                    psi,
                    step.action,
                    coeff=complex(step.coeff_real),
                    dt=dt,
                    tol=self.coefficient_tolerance,
                )
        return psi


__all__ = ["CompiledAnsatzExecutor"]
