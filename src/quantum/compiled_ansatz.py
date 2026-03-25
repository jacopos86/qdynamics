"""Compiled ansatz execution helpers (exyz convention).

This module provides a wrapper executor that prepares ansatz states from
``AnsatzTerm``-like inputs using shared compiled Pauli actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    RotationTermSpec,
    build_parameter_layout,
    iter_runtime_rotation_terms,
)
from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_compiled_pauli,
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
        parameterization_layout: AnsatzParameterLayout | None = None,
    ):
        self.coefficient_tolerance = float(coefficient_tolerance)
        self.ignore_identity = bool(ignore_identity)
        self.sort_terms = bool(sort_terms)
        self.parameterization_mode = str(parameterization_mode)
        self.terms = list(terms)

        self.pauli_action_cache = (
            pauli_action_cache if pauli_action_cache is not None else {}
        )
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
            if parameterization_layout is None:
                plan = self._compile_polynomial_plan(
                    getattr(term, "polynomial"),
                    label=str(block.candidate_label),
                    runtime_start=int(block.runtime_start),
                )
            else:
                plan = self._compile_rotation_specs(
                    tuple(block.terms),
                    poly=getattr(term, "polynomial"),
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

    def _compile_rotation_specs(
        self,
        specs: Sequence[RotationTermSpec],
        *,
        poly: Any,
        label: str,
        runtime_start: int,
    ) -> CompiledPolynomialRotationPlan:
        if len(specs) == 0:
            poly_terms = list(poly.return_polynomial())
            nq = int(poly_terms[0].nqubit()) if poly_terms else None
            return CompiledPolynomialRotationPlan(
                nq=nq,
                label=str(label),
                steps=tuple(),
                runtime_start=int(runtime_start),
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

        return CompiledPolynomialRotationPlan(
            nq=int(specs[0].nq),
            label=str(label),
            steps=tuple(compiled_steps),
            runtime_start=int(runtime_start),
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
        return self._compile_rotation_specs(
            tuple(block.terms) if block is not None else tuple(),
            poly=poly,
            label=str(label),
            runtime_start=int(runtime_start),
        )

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


__all__ = ["CompiledAnsatzExecutor"]
