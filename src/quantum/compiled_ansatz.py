"""Compiled ansatz execution helpers (exyz convention).

This module provides a wrapper executor that prepares ansatz states from
``AnsatzTerm``-like inputs using shared compiled Pauli actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

from src.quantum.ansatz_parameterization import AnsatzParameterLayout, build_parameter_layout
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
    label: str
    steps: tuple[CompiledRotationStep, ...]
    runtime_start: int

    @property
    def runtime_count(self) -> int:
        return int(len(self.steps))

    @property
    def runtime_stop(self) -> int:
        return int(self.runtime_start + len(self.steps))


class CompiledAnsatzExecutor:
    """Compiled-action ansatz executor for lists of AnsatzTerm-like objects."""

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
    ):
        self.coefficient_tolerance = float(coefficient_tolerance)
        self.ignore_identity = bool(ignore_identity)
        self.sort_terms = bool(sort_terms)
        self.parameterization_mode = str(parameterization_mode)
        self.terms = list(terms)

        self.pauli_action_cache = (
            pauli_action_cache if pauli_action_cache is not None else {}
        )
        self.layout: AnsatzParameterLayout = build_parameter_layout(
            self.terms,
            ignore_identity=self.ignore_identity,
            coefficient_tolerance=self.coefficient_tolerance,
            sort_terms=self.sort_terms,
        )
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
            plan = self._compile_polynomial_plan(
                getattr(term, "polynomial"),
                label=str(block.candidate_label),
                runtime_start=int(block.runtime_start),
            )
            if plan.nq is not None:
                if self.nq is None:
                    self.nq = int(plan.nq)
                elif int(plan.nq) != int(self.nq):
                    raise ValueError(
                        f"Inconsistent ansatz qubit counts: saw {plan.nq} after {self.nq}."
                    )
            self._plans.append(plan)

    _MATH_COMPILE_POLYNOMIAL_PLAN = (
        r"H=\\sum_j c_j P_j,\\ \\text{ordered as in runtime termwise execution},"
        r"\\ \\text{skip }|c_j|<\\epsilon,\\ \\text{optional skip }I"
    )

    def _compile_polynomial_plan(
        self,
        poly: Any,
        *,
        label: str,
        runtime_start: int,
    ) -> CompiledPolynomialRotationPlan:
        layout_single = build_parameter_layout(
            [type("_TermCarrier", (), {"label": label, "polynomial": poly})()],
            ignore_identity=self.ignore_identity,
            coefficient_tolerance=self.coefficient_tolerance,
            sort_terms=self.sort_terms,
        )
        block = layout_single.blocks[0] if layout_single.blocks else None
        if block is None or len(block.terms) == 0:
            poly_terms = list(poly.return_polynomial())
            nq = int(poly_terms[0].nqubit()) if poly_terms else None
            return CompiledPolynomialRotationPlan(
                nq=nq,
                label=str(label),
                steps=tuple(),
                runtime_start=int(runtime_start),
            )

        compiled_steps: list[CompiledRotationStep] = []
        for spec in block.terms:
            action = self.pauli_action_cache.get(spec.pauli_exyz)
            if action is None:
                action = compile_pauli_action_exyz(spec.pauli_exyz, int(spec.nq))
                self.pauli_action_cache[spec.pauli_exyz] = action
            compiled_steps.append(
                CompiledRotationStep(coeff_real=float(spec.coeff_real), action=action)
            )

        return CompiledPolynomialRotationPlan(
            nq=int(block.terms[0].nq),
            label=str(label),
            steps=tuple(compiled_steps),
            runtime_start=int(runtime_start),
        )

    _MATH_PREPARE_STATE = (
        r"|\\psi(\\vartheta)\\rangle=\\prod_k \\prod_j \\exp(-i\\vartheta_{k,j} c_{k,j}P_{k,j})|\\psi_{\\mathrm{ref}}\\rangle"
    )

    def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        """Prepare ansatz state using compiled Pauli actions."""
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

        psi = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
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
