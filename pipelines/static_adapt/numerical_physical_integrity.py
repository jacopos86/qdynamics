"""Reporting-only numerical and physical integrity receipts for static ADAPT.

The helpers in this module run only after an optimizer/controller has completed.
They inspect already-produced states and typed transition records; they never
provide data to selection, admission, refit, pruning, or stopping decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.contracts.problem import (
    FixedCountConstraint,
    ResolvedProblemContext,
    TruncationConstraint,
    WeightedChargeConstraint,
)
from pipelines.static_adapt.estimator_call_ledger import (
    projective_state_fingerprint,
)
from pipelines.static_adapt.sr_snake.contracts import (
    SRRunResult,
    SerializableContract,
)
from src.quantum.hartree_fock_reference_state import mode_index
from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site
from src.quantum.operator_pools.boson_chains import (
    boson_chain_legal_basis_indices,
)


NUMERICAL_PHYSICAL_INTEGRITY_SCHEMA = (
    "paper_i_numerical_physical_integrity_v1"
)
ENERGY_TRANSITION_INTEGRITY_SCHEMA = (
    "paper_i_accepted_energy_transition_integrity_v1"
)
SECTOR_DIAGNOSTIC_POLICY = (
    "reporting_only_post_controller_sector_probability_v1"
)
RA_INTEGRITY_DERIVATION = (
    "post_controller_typed_result_and_signed_terminal_checkpoint_v1"
)
APPEND_INTEGRITY_DERIVATION = (
    "post_controller_typed_result_and_terminal_statevector_v1"
)
SECTOR_LEAK_THRESHOLD = 1.0e-8
APPEND_NONWORSENING_ABSOLUTE_TOLERANCE = 0.0
RESUMED_RA_NONWORSENING_ABSOLUTE_TOLERANCE = 0.0


def _clamped_probability(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite probability.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite probability.")
    tolerance = 1.0e-12
    if result < -tolerance or result > 1.0 + tolerance:
        raise ValueError(f"{name} must lie in [0, 1].")
    return float(min(1.0, max(0.0, result)))


def _fixed_count_qubits(
    context: ResolvedProblemContext,
    quantity: str,
    scope: str,
) -> tuple[int, ...]:
    quantity_key = str(quantity)
    request = context.request
    n_sites = int(request.num_sites)
    if quantity_key == "n_up":
        return tuple(
            mode_index(
                site,
                0,
                n_sites=n_sites,
                indexing=str(request.ordering),
            )
            for site in range(n_sites)
        )
    if quantity_key == "n_dn":
        return tuple(
            mode_index(
                site,
                1,
                n_sites=n_sites,
                indexing=str(request.ordering),
            )
            for site in range(n_sites)
        )
    if quantity_key == "n_f":
        block = context.layout.block("fermion")
        if block is not None:
            return tuple(
                range(int(block.start_qubit), int(block.stop_qubit))
            )
        return tuple(
            range(
                int(
                    context.layout.fermion_qubits
                    or context.layout.total_qubits
                )
            )
        )
    scope_key = str(scope).replace("_register", "")
    block = context.layout.block(scope_key)
    if block is not None:
        return tuple(
            range(int(block.start_qubit), int(block.stop_qubit))
        )
    return tuple(range(int(context.layout.total_qubits)))


def _truncation_constraint_spec(
    context: ResolvedProblemContext,
    constraint: TruncationConstraint,
) -> dict[str, Any]:
    scope = str(constraint.scope)
    scope_key = scope.replace("_register", "")
    block = context.layout.block(scope_key)
    if block is None:
        if scope_key == "full":
            start_qubit = 0
            stop_qubit = int(context.layout.total_qubits)
        else:
            raise ValueError(
                f"No layout block found for truncation scope {scope!r}."
            )
    else:
        start_qubit = int(block.start_qubit)
        stop_qubit = int(block.stop_qubit)

    qubits = tuple(range(start_qubit, stop_qubit))
    register_size = len(qubits)
    if register_size <= 0:
        raise ValueError(f"Truncation scope {scope!r} has no qubits.")
    max_local_occupancy = int(constraint.max_local_occupancy)
    encoding = str(
        getattr(context.layout, "boson_encoding", None)
        or getattr(context.request, "boson_encoding", "binary")
    )
    bits_per_site = int(
        boson_qubits_per_site(max_local_occupancy, encoding)
    )
    if bits_per_site <= 0 or register_size % bits_per_site != 0:
        raise ValueError(
            "Truncation register size is incompatible with the boson "
            f"encoding: scope={scope!r}, register_size={register_size}, "
            f"bits_per_site={bits_per_site}."
        )
    num_sites = int(register_size // bits_per_site)
    legal = frozenset(
        int(value)
        for value in boson_chain_legal_basis_indices(
            num_sites=num_sites,
            n_ph_max=max_local_occupancy,
            boson_encoding=encoding,
        ).tolist()
    )
    return {
        "quantity": str(constraint.quantity),
        "scope": scope,
        "max_local_occupancy": max_local_occupancy,
        "qubits": list(qubits),
        "start_qubit": start_qubit,
        "stop_qubit": stop_qubit,
        "num_sites": num_sites,
        "boson_encoding": encoding,
        "bits_per_site": bits_per_site,
        "legal_basis_count": len(legal),
        "_register_mask": (1 << register_size) - 1,
        "_legal_register_indices": legal,
    }


def sector_probability(
    context: ResolvedProblemContext,
    psi: np.ndarray,
) -> dict[str, Any]:
    """Return reporting-only fixed-sector and truncation probabilities."""

    psi_arr = np.asarray(psi, dtype=complex).reshape(-1)
    expected_dimension = 1 << int(context.layout.total_qubits)
    if int(psi_arr.size) != expected_dimension:
        raise ValueError(
            "Statevector dimension does not match the resolved problem "
            f"layout: {psi_arr.size} != {expected_dimension}."
        )
    if not bool(
        np.all(np.isfinite(psi_arr.real))
        and np.all(np.isfinite(psi_arr.imag))
    ):
        raise ValueError(
            "Cannot compute sector diagnostics for a non-finite state."
        )
    norm = float(np.vdot(psi_arr, psi_arr).real)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(
            "Cannot compute sector diagnostics for a zero-norm or "
            "non-finite state."
        )
    probs = np.abs(psi_arr) ** 2 / norm
    constraints = tuple(context.sector.constraints)
    fixed_constraints = [
        constraint
        for constraint in constraints
        if isinstance(
            constraint,
            (FixedCountConstraint, WeightedChargeConstraint),
        )
    ]
    truncation_constraints = [
        constraint
        for constraint in constraints
        if isinstance(constraint, TruncationConstraint)
    ]
    if not fixed_constraints and not truncation_constraints:
        return {
            "sector_probability": 1.0,
            "sector_leak_probability": 0.0,
            "fixed_count_sector_probability": 1.0,
            "fixed_count_sector_leak_probability": 0.0,
            "sector_leak_flag": False,
            "sector_leak_threshold": SECTOR_LEAK_THRESHOLD,
            "constraints_evaluated": [],
            "fixed_count_constraints_evaluated": [],
            "truncation_constraints_evaluated": [],
            "boson_subspace_diagnostics": None,
            "boson_legal_probability_min": None,
            "boson_illegal_probability_max": None,
            "boson_truncation_leak_flag": False,
            "policy": "no_fixed_count_constraints",
        }

    evaluated: list[dict[str, Any]] = []
    constraint_specs: list[
        tuple[tuple[int, ...], int, str, str]
    ] = []
    for constraint in fixed_constraints:
        qubits = _fixed_count_qubits(
            context,
            constraint.quantity,
            constraint.scope,
        )
        target = int(constraint.value)
        quantity = str(constraint.quantity)
        scope = str(constraint.scope)
        constraint_specs.append((qubits, target, quantity, scope))
        evaluated.append(
            {
                "quantity": quantity,
                "scope": scope,
                "value": target,
                "qubits": list(qubits),
            }
        )

    truncation_specs = [
        _truncation_constraint_spec(context, constraint)
        for constraint in truncation_constraints
    ]
    truncation_good_probs = [0.0 for _ in truncation_specs]

    fixed_good_prob = 0.0
    joint_good_prob = 0.0
    for basis_index, prob in enumerate(probs):
        prob_f = float(prob)
        if prob_f <= 0.0:
            continue
        fixed_ok = True
        for qubits, target, _quantity, _scope in constraint_specs:
            count = sum(
                (int(basis_index) >> int(qubit)) & 1
                for qubit in qubits
            )
            if count != target:
                fixed_ok = False
                break
        if fixed_ok:
            fixed_good_prob += prob_f
        truncation_ok = True
        for index, spec in enumerate(truncation_specs):
            register_index = (
                int(basis_index) >> int(spec["start_qubit"])
            ) & int(spec["_register_mask"])
            if register_index in spec["_legal_register_indices"]:
                truncation_good_probs[index] += prob_f
            else:
                truncation_ok = False
        if fixed_ok and truncation_ok:
            joint_good_prob += prob_f

    truncation_evaluated: list[dict[str, Any]] = []
    for spec, legal_prob in zip(
        truncation_specs,
        truncation_good_probs,
        strict=True,
    ):
        legal = _clamped_probability(
            legal_prob,
            name="boson legal probability",
        )
        illegal = _clamped_probability(
            1.0 - legal,
            name="boson illegal probability",
        )
        public = {
            key: value
            for key, value in spec.items()
            if not key.startswith("_")
        }
        public.update(
            {
                "legal_probability": legal,
                "illegal_probability": illegal,
            }
        )
        truncation_evaluated.append(public)

    fixed_probability = _clamped_probability(
        fixed_good_prob if constraint_specs else 1.0,
        name="fixed-count sector probability",
    )
    fixed_leak = _clamped_probability(
        1.0 - fixed_probability,
        name="fixed-count sector leak probability",
    )
    joint_probability = _clamped_probability(
        joint_good_prob,
        name="joint sector probability",
    )
    joint_leak = _clamped_probability(
        1.0 - joint_probability,
        name="joint sector leak probability",
    )
    boson_legal_probability_min = (
        min(
            float(item["legal_probability"])
            for item in truncation_evaluated
        )
        if truncation_evaluated
        else None
    )
    boson_illegal_probability_max = (
        max(
            float(item["illegal_probability"])
            for item in truncation_evaluated
        )
        if truncation_evaluated
        else None
    )
    boson_truncation_leak_flag = bool(
        boson_illegal_probability_max is not None
        and boson_illegal_probability_max > SECTOR_LEAK_THRESHOLD
    )
    if constraint_specs and truncation_specs:
        policy = (
            "diagnostic_only_fixed_count_and_truncation_probability"
        )
    elif truncation_specs:
        policy = "diagnostic_only_truncation_probability"
    else:
        policy = "diagnostic_only_fixed_count_probability"
    boson_subspace_diagnostics = None
    if truncation_evaluated:
        boson_subspace_diagnostics = {
            "policy": "reporting_only_after_optimizer",
            "boson_legal_probability_min": (
                boson_legal_probability_min
            ),
            "boson_illegal_probability_max": (
                boson_illegal_probability_max
            ),
            "constraints_evaluated": truncation_evaluated,
        }
    return {
        "sector_probability": joint_probability,
        "sector_leak_probability": joint_leak,
        "fixed_count_sector_probability": fixed_probability,
        "fixed_count_sector_leak_probability": fixed_leak,
        "sector_leak_flag": bool(
            joint_leak > SECTOR_LEAK_THRESHOLD
        ),
        "sector_leak_threshold": SECTOR_LEAK_THRESHOLD,
        "constraints_evaluated": [
            *evaluated,
            *truncation_evaluated,
        ],
        "fixed_count_constraints_evaluated": evaluated,
        "truncation_constraints_evaluated": truncation_evaluated,
        "boson_subspace_diagnostics": boson_subspace_diagnostics,
        "boson_legal_probability_min": boson_legal_probability_min,
        "boson_illegal_probability_max": (
            boson_illegal_probability_max
        ),
        "boson_truncation_leak_flag": (
            boson_truncation_leak_flag
        ),
        "policy": policy,
    }


@dataclass(frozen=True)
class AcceptedEnergyTransitionIntegrityReceipt(SerializableContract):
    schema: str
    controller_round: int
    energy_before: float
    energy_after: float
    absolute_tolerance: float
    comparison_semantics: str
    nonincrease_passed: bool
    typed_rollback_receipt: Mapping[str, Any] | None
    gate_passed: bool

    def __post_init__(self) -> None:
        if self.schema != ENERGY_TRANSITION_INTEGRITY_SCHEMA:
            raise ValueError(
                "Unknown accepted-energy transition integrity schema."
            )
        if int(self.controller_round) < 1:
            raise ValueError(
                "Accepted-energy transition round must be positive."
            )
        before = float(self.energy_before)
        after = float(self.energy_after)
        tolerance = float(self.absolute_tolerance)
        if (
            not math.isfinite(before)
            or not math.isfinite(after)
            or not math.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError(
                "Accepted-energy transition values must be finite and the "
                "tolerance must be nonnegative."
            )
        if not str(self.comparison_semantics).strip():
            raise ValueError(
                "Accepted-energy comparison semantics must be explicit."
            )
        expected_nonincrease = bool(after <= before + tolerance)
        if self.nonincrease_passed is not expected_nonincrease:
            raise ValueError(
                "Accepted-energy nonincrease flag disagrees with its values."
            )
        rollback_present = self.typed_rollback_receipt is not None
        if rollback_present and not isinstance(
            self.typed_rollback_receipt,
            Mapping,
        ):
            raise TypeError("Typed rollback evidence must be a mapping.")
        if self.gate_passed is not bool(
            expected_nonincrease or rollback_present
        ):
            raise ValueError(
                "Accepted-energy gate flag disagrees with nonincrease or "
                "typed rollback evidence."
            )


@dataclass(frozen=True)
class NumericalPhysicalIntegrityReceipt(SerializableContract):
    schema: str
    method: str
    derivation_policy: str
    reporting_only: bool
    controller_decision_influence: bool
    finite_values_passed: bool
    checked_energy_value_count: int
    checked_parameter_value_count: int
    nonfinite_value_paths: tuple[str, ...]
    sector_diagnostic_policy: str
    state_fingerprint: str
    sector_leak_threshold: float
    fixed_count_sector_probability: float
    fixed_count_sector_leak_probability: float
    sector_leak_flag: bool
    boson_legal_probability_min: float | None
    boson_illegal_probability_max: float | None
    boson_truncation_leak_flag: bool
    accepted_energy_transitions: tuple[
        AcceptedEnergyTransitionIntegrityReceipt,
        ...,
    ]
    accepted_energy_integrity_passed: bool
    integrity_passed: bool

    def __post_init__(self) -> None:
        if self.schema != NUMERICAL_PHYSICAL_INTEGRITY_SCHEMA:
            raise ValueError(
                "Unknown numerical/physical integrity schema."
            )
        if not str(self.method).strip() or not str(
            self.derivation_policy
        ).strip():
            raise ValueError(
                "Integrity method and derivation policy are required."
            )
        if self.reporting_only is not True:
            raise ValueError(
                "Numerical/physical integrity must remain reporting-only."
            )
        if self.controller_decision_influence is not False:
            raise ValueError(
                "Numerical/physical integrity cannot influence controller "
                "decisions."
            )
        if (
            int(self.checked_energy_value_count) < 0
            or int(self.checked_parameter_value_count) < 0
        ):
            raise ValueError(
                "Integrity checked-value counts cannot be negative."
            )
        if self.finite_values_passed is not (
            len(self.nonfinite_value_paths) == 0
        ):
            raise ValueError(
                "Finite-value integrity flag disagrees with nonfinite paths."
            )
        if not str(self.sector_diagnostic_policy).strip():
            raise ValueError("Sector diagnostic policy is required.")
        if not str(self.state_fingerprint).strip():
            raise ValueError("State fingerprint is required.")
        threshold = float(self.sector_leak_threshold)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "Sector leak threshold must be finite and nonnegative."
            )
        fixed_probability = _clamped_probability(
            self.fixed_count_sector_probability,
            name="fixed-count sector probability",
        )
        fixed_leak = _clamped_probability(
            self.fixed_count_sector_leak_probability,
            name="fixed-count sector leak probability",
        )
        if not math.isclose(
            fixed_probability + fixed_leak,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise ValueError(
                "Fixed-count sector probability and leak do not close."
            )
        if self.boson_legal_probability_min is None:
            if (
                self.boson_illegal_probability_max is not None
                or self.boson_truncation_leak_flag
            ):
                raise ValueError(
                    "Absent boson truncation constraints cannot report "
                    "boson leakage."
                )
        else:
            legal = _clamped_probability(
                self.boson_legal_probability_min,
                name="boson legal probability",
            )
            illegal = _clamped_probability(
                self.boson_illegal_probability_max,
                name="boson illegal probability",
            )
            if not math.isclose(
                legal + illegal,
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ):
                raise ValueError(
                    "Boson legal and illegal probabilities do not close."
                )
            if self.boson_truncation_leak_flag is not bool(
                illegal > threshold
            ):
                raise ValueError(
                    "Boson truncation leak flag disagrees with its "
                    "probability."
                )
        energy_passed = all(
            row.gate_passed for row in self.accepted_energy_transitions
        )
        if self.accepted_energy_integrity_passed is not energy_passed:
            raise ValueError(
                "Accepted-energy integrity flag disagrees with its "
                "transition receipts."
            )
        expected_integrity = bool(
            self.finite_values_passed
            and not self.sector_leak_flag
            and not self.boson_truncation_leak_flag
            and self.accepted_energy_integrity_passed
        )
        if self.integrity_passed is not expected_integrity:
            raise ValueError(
                "Overall numerical/physical integrity flag does not close."
            )


def _finite_value_audit(
    values: Sequence[tuple[str, Any]],
) -> tuple[int, tuple[str, ...]]:
    nonfinite: list[str] = []
    for path, value in values:
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            nonfinite.append(str(path))
    return len(values), tuple(nonfinite)


def numerical_physical_integrity_from_mapping(
    payload: Mapping[str, Any],
) -> NumericalPhysicalIntegrityReceipt:
    """Rehydrate and validate one serialized integrity receipt."""

    if not isinstance(payload, Mapping):
        raise TypeError(
            "Numerical/physical integrity payload must be a mapping."
        )
    raw_transitions = payload.get("accepted_energy_transitions")
    if not isinstance(raw_transitions, Sequence) or isinstance(
        raw_transitions,
        (str, bytes),
    ):
        raise TypeError(
            "Integrity accepted-energy transitions must be a sequence."
        )
    transitions: list[
        AcceptedEnergyTransitionIntegrityReceipt
    ] = []
    for index, raw in enumerate(raw_transitions):
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Integrity transition[{index}] must be a mapping."
            )
        transitions.append(
            AcceptedEnergyTransitionIntegrityReceipt(
                schema=str(raw.get("schema", "")),
                controller_round=int(raw.get("controller_round", 0)),
                energy_before=float(raw.get("energy_before")),
                energy_after=float(raw.get("energy_after")),
                absolute_tolerance=float(
                    raw.get("absolute_tolerance")
                ),
                comparison_semantics=str(
                    raw.get("comparison_semantics", "")
                ),
                nonincrease_passed=raw.get(
                    "nonincrease_passed"
                ),
                typed_rollback_receipt=raw.get(
                    "typed_rollback_receipt"
                ),
                gate_passed=raw.get("gate_passed"),
            )
        )
    nonfinite_raw = payload.get("nonfinite_value_paths")
    if not isinstance(nonfinite_raw, Sequence) or isinstance(
        nonfinite_raw,
        (str, bytes),
    ):
        raise TypeError(
            "Integrity nonfinite-value paths must be a sequence."
        )
    return NumericalPhysicalIntegrityReceipt(
        schema=str(payload.get("schema", "")),
        method=str(payload.get("method", "")),
        derivation_policy=str(payload.get("derivation_policy", "")),
        reporting_only=payload.get("reporting_only"),
        controller_decision_influence=payload.get(
            "controller_decision_influence"
        ),
        finite_values_passed=payload.get("finite_values_passed"),
        checked_energy_value_count=int(
            payload.get("checked_energy_value_count", -1)
        ),
        checked_parameter_value_count=int(
            payload.get("checked_parameter_value_count", -1)
        ),
        nonfinite_value_paths=tuple(
            str(value) for value in nonfinite_raw
        ),
        sector_diagnostic_policy=str(
            payload.get("sector_diagnostic_policy", "")
        ),
        state_fingerprint=str(payload.get("state_fingerprint", "")),
        sector_leak_threshold=float(
            payload.get("sector_leak_threshold")
        ),
        fixed_count_sector_probability=float(
            payload.get("fixed_count_sector_probability")
        ),
        fixed_count_sector_leak_probability=float(
            payload.get("fixed_count_sector_leak_probability")
        ),
        sector_leak_flag=payload.get("sector_leak_flag"),
        boson_legal_probability_min=(
            None
            if payload.get("boson_legal_probability_min") is None
            else float(payload["boson_legal_probability_min"])
        ),
        boson_illegal_probability_max=(
            None
            if payload.get("boson_illegal_probability_max") is None
            else float(payload["boson_illegal_probability_max"])
        ),
        boson_truncation_leak_flag=payload.get(
            "boson_truncation_leak_flag"
        ),
        accepted_energy_transitions=tuple(transitions),
        accepted_energy_integrity_passed=payload.get(
            "accepted_energy_integrity_passed"
        ),
        integrity_passed=payload.get("integrity_passed"),
    )


def _energy_transition_receipt(
    *,
    controller_round: int,
    energy_before: Any,
    energy_after: Any,
    absolute_tolerance: Any,
    comparison_semantics: str,
    typed_rollback_receipt: Mapping[str, Any] | None = None,
) -> AcceptedEnergyTransitionIntegrityReceipt:
    before = float(energy_before)
    after = float(energy_after)
    tolerance = float(absolute_tolerance)
    nonincrease = bool(after <= before + tolerance)
    return AcceptedEnergyTransitionIntegrityReceipt(
        schema=ENERGY_TRANSITION_INTEGRITY_SCHEMA,
        controller_round=int(controller_round),
        energy_before=before,
        energy_after=after,
        absolute_tolerance=tolerance,
        comparison_semantics=str(comparison_semantics),
        nonincrease_passed=nonincrease,
        typed_rollback_receipt=typed_rollback_receipt,
        gate_passed=bool(
            nonincrease or typed_rollback_receipt is not None
        ),
    )


def _terminal_checkpoint(
    finalization: Mapping[str, Any],
) -> Mapping[str, Any]:
    checkpoint = finalization.get(
        "terminal_active_prefix_checkpoint"
    )
    if isinstance(checkpoint, Mapping):
        return checkpoint
    continuation = finalization.get("continuation")
    if isinstance(continuation, Mapping):
        checkpoint = continuation.get(
            "terminal_active_prefix_checkpoint"
        )
        if isinstance(checkpoint, Mapping):
            return checkpoint
    raise RuntimeError(
        "RA integrity projection requires the signed terminal active-prefix "
        "checkpoint."
    )


def build_ra_numerical_physical_integrity(
    *,
    run: SRRunResult,
    finalization: Mapping[str, Any],
) -> NumericalPhysicalIntegrityReceipt:
    """Project G9 evidence after one completed RA controller run."""

    if not isinstance(run, SRRunResult):
        raise TypeError("RA integrity requires a typed SRRunResult.")
    checkpoint = _terminal_checkpoint(finalization)
    checkpoint_fingerprint = str(
        checkpoint.get("projective_state_fingerprint", "")
    )
    if (
        not checkpoint_fingerprint
        or checkpoint_fingerprint
        != run.final_state.projective_state_fingerprint
    ):
        raise RuntimeError(
            "RA terminal checkpoint and typed final-state fingerprints "
            "disagree."
        )
    state_sector_raw = checkpoint.get("state_sector_contract")
    if not isinstance(state_sector_raw, Mapping):
        raise RuntimeError(
            "RA terminal checkpoint omitted its state-sector contract."
        )
    fixed_probability_raw = state_sector_raw.get(
        "joint_target_sector_probability"
    )
    fixed_probability = (
        1.0
        if fixed_probability_raw is None
        else _clamped_probability(
            fixed_probability_raw,
            name="RA fixed-count sector probability",
        )
    )
    fixed_leak = _clamped_probability(
        1.0 - fixed_probability,
        name="RA fixed-count sector leak probability",
    )
    checkpoint_fixed_probability = checkpoint.get(
        "fixed_spin_sector_probability"
    )
    if checkpoint_fixed_probability is not None and not math.isclose(
        _clamped_probability(
            checkpoint_fixed_probability,
            name="RA checkpoint fixed-spin sector probability",
        ),
        fixed_probability,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise RuntimeError(
            "RA terminal fixed-spin and state-sector probabilities disagree."
        )
    boson_legal_raw = checkpoint.get(
        "boson_legal_codeword_probability"
    )
    boson_legal = (
        None
        if boson_legal_raw is None
        else _clamped_probability(
            boson_legal_raw,
            name="RA boson legal probability",
        )
    )
    boson_illegal = (
        None
        if boson_legal is None
        else _clamped_probability(
            1.0 - boson_legal,
            name="RA boson illegal probability",
        )
    )
    checkpoint_boson_illegal = checkpoint.get(
        "boson_illegal_codeword_probability"
    )
    if checkpoint_boson_illegal is not None:
        observed_illegal = _clamped_probability(
            checkpoint_boson_illegal,
            name="RA checkpoint boson illegal probability",
        )
        if boson_illegal is None or not math.isclose(
            observed_illegal,
            boson_illegal,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise RuntimeError(
                "RA terminal boson legal and illegal probabilities "
                "disagree."
            )
    threshold = SECTOR_LEAK_THRESHOLD
    sector_leak_flag = bool(
        state_sector_raw.get("passed") is not True
        or fixed_leak > threshold
        or (
            boson_illegal is not None
            and boson_illegal > threshold
        )
    )
    boson_leak_flag = bool(
        boson_illegal is not None
        and boson_illegal > threshold
    )

    energy_values: list[tuple[str, Any]] = [
        ("run.stop.accepted_energy", run.stop.accepted_energy),
        (
            "run.canonical_reporting.exact_same_cutoff_energy",
            run.canonical_reporting.exact_same_cutoff_energy,
        ),
    ]
    for field_name in (
        "exact_target_energy",
        "exact_absolute_tolerance",
        "exact_observed_absolute_difference",
    ):
        value = getattr(run.stop, field_name)
        if value is not None:
            energy_values.append((f"run.stop.{field_name}", value))
    parameter_values: list[tuple[str, Any]] = []
    for index, state in enumerate(run.accepted_trajectory):
        energy_values.append(
            (
                f"run.accepted_trajectory[{index}].energy",
                state.energy,
            )
        )
        parameter_values.extend(
            (
                (
                    f"run.accepted_trajectory[{index}]."
                    f"logical_parameters[{parameter_index}]",
                    value,
                )
                for parameter_index, value in enumerate(
                    state.logical_parameters
                )
            )
        )
        parameter_values.extend(
            (
                (
                    f"run.accepted_trajectory[{index}]."
                    f"runtime_parameters[{parameter_index}]",
                    value,
                )
                for parameter_index, value in enumerate(
                    state.runtime_parameters
                )
            )
        )

    transitions: list[
        AcceptedEnergyTransitionIntegrityReceipt
    ] = []
    for index, transition in enumerate(run.accepted_transitions):
        energy_values.extend(
            (
                (
                    f"run.accepted_transitions[{index}].energy_before",
                    transition.energy_before,
                ),
                (
                    f"run.accepted_transitions[{index}].energy_after",
                    transition.energy_after,
                ),
            )
        )
        typed_tolerance = getattr(
            transition,
            "non_worsening_absolute_tolerance",
            None,
        )
        tolerance = (
            RESUMED_RA_NONWORSENING_ABSOLUTE_TOLERANCE
            if typed_tolerance is None
            else float(typed_tolerance)
        )
        transitions.append(
            _energy_transition_receipt(
                controller_round=int(transition.controller_round),
                energy_before=transition.energy_before,
                energy_after=transition.energy_after,
                absolute_tolerance=tolerance,
                comparison_semantics=(
                    "typed_transition_non_worsening_absolute_tolerance_v1"
                    if typed_tolerance is not None
                    else (
                        "authenticated_resume_reporting_absolute_"
                        "tolerance_v1"
                    )
                ),
            )
        )
    for index, replay in enumerate(run.scientific_replay):
        energy_values.extend(
            (
                (
                    f"run.scientific_replay[{index}]."
                    "energy_before_refit",
                    replay.energy_before_refit,
                ),
                (
                    f"run.scientific_replay[{index}]."
                    "accepted_refit.final_energy",
                    replay.accepted_refit.final_energy,
                ),
            )
        )

    energy_count, nonfinite_energy = _finite_value_audit(
        energy_values
    )
    parameter_count, nonfinite_parameters = _finite_value_audit(
        parameter_values
    )
    nonfinite = (*nonfinite_energy, *nonfinite_parameters)
    energy_integrity = all(row.gate_passed for row in transitions)
    finite_passed = not nonfinite
    integrity_passed = bool(
        finite_passed
        and not sector_leak_flag
        and not boson_leak_flag
        and energy_integrity
    )
    return NumericalPhysicalIntegrityReceipt(
        schema=NUMERICAL_PHYSICAL_INTEGRITY_SCHEMA,
        method="ra_adapt",
        derivation_policy=RA_INTEGRITY_DERIVATION,
        reporting_only=True,
        controller_decision_influence=False,
        finite_values_passed=finite_passed,
        checked_energy_value_count=energy_count,
        checked_parameter_value_count=parameter_count,
        nonfinite_value_paths=tuple(nonfinite),
        sector_diagnostic_policy=(
            "signed_terminal_checkpoint_fixed_sector_and_boson_"
            "codeword_probability_v1"
        ),
        state_fingerprint=checkpoint_fingerprint,
        sector_leak_threshold=threshold,
        fixed_count_sector_probability=fixed_probability,
        fixed_count_sector_leak_probability=fixed_leak,
        sector_leak_flag=sector_leak_flag,
        boson_legal_probability_min=boson_legal,
        boson_illegal_probability_max=boson_illegal,
        boson_truncation_leak_flag=boson_leak_flag,
        accepted_energy_transitions=tuple(transitions),
        accepted_energy_integrity_passed=energy_integrity,
        integrity_passed=integrity_passed,
    )


def build_append_numerical_physical_integrity(
    *,
    problem: ResolvedProblemContext,
    final_state: np.ndarray,
    history: Sequence[Mapping[str, Any]],
    logical_parameters: Sequence[float],
    runtime_parameters: Sequence[float],
    final_energy: float,
) -> NumericalPhysicalIntegrityReceipt:
    """Project G9 evidence after one completed conventional Append run."""

    diagnostic = sector_probability(problem, final_state)
    energy_values: list[tuple[str, Any]] = [
        ("result_payload.final_energy", final_energy),
    ]
    parameter_values: list[tuple[str, Any]] = [
        (
            f"result_payload.logical_theta[{index}]",
            value,
        )
        for index, value in enumerate(logical_parameters)
    ]
    parameter_values.extend(
        (
            f"result_payload.runtime_theta[{index}]",
            value,
        )
        for index, value in enumerate(runtime_parameters)
    )
    transitions: list[
        AcceptedEnergyTransitionIntegrityReceipt
    ] = []
    for index, row in enumerate(history):
        if not isinstance(row, Mapping):
            raise TypeError(
                f"Append history[{index}] must be a mapping."
            )
        energy_before = row.get("energy_before")
        energy_after = row.get("energy_after")
        energy_values.extend(
            (
                (
                    f"result_payload.history[{index}].energy_before",
                    energy_before,
                ),
                (
                    f"result_payload.history[{index}].energy_after",
                    energy_after,
                ),
            )
        )
        refit = row.get("accepted_refit")
        if isinstance(refit, Mapping):
            if refit.get("final_energy") is not None:
                energy_values.append(
                    (
                        f"result_payload.history[{index}]."
                        "accepted_refit.final_energy",
                        refit["final_energy"],
                    )
                )
            for field_name in (
                "origin_logical_theta",
                "origin_runtime_theta",
                "final_logical_theta",
                "final_runtime_theta",
            ):
                raw_values = refit.get(field_name, ())
                if isinstance(raw_values, Sequence) and not isinstance(
                    raw_values,
                    (str, bytes),
                ):
                    parameter_values.extend(
                        (
                            f"result_payload.history[{index}]."
                            f"accepted_refit.{field_name}[{value_index}]",
                            value,
                        )
                        for value_index, value in enumerate(raw_values)
                    )
        transitions.append(
            _energy_transition_receipt(
                controller_round=int(
                    row.get("controller_round", index + 1)
                ),
                energy_before=energy_before,
                energy_after=energy_after,
                absolute_tolerance=(
                    APPEND_NONWORSENING_ABSOLUTE_TOLERANCE
                ),
                comparison_semantics=(
                    "append_zero_angle_origin_reporting_absolute_"
                    "tolerance_v1"
                ),
            )
        )

    energy_count, nonfinite_energy = _finite_value_audit(
        energy_values
    )
    parameter_count, nonfinite_parameters = _finite_value_audit(
        parameter_values
    )
    nonfinite = (*nonfinite_energy, *nonfinite_parameters)
    energy_integrity = all(row.gate_passed for row in transitions)
    finite_passed = not nonfinite
    sector_leak_flag = bool(diagnostic["sector_leak_flag"])
    boson_leak_flag = bool(
        diagnostic["boson_truncation_leak_flag"]
    )
    integrity_passed = bool(
        finite_passed
        and not sector_leak_flag
        and not boson_leak_flag
        and energy_integrity
    )
    return NumericalPhysicalIntegrityReceipt(
        schema=NUMERICAL_PHYSICAL_INTEGRITY_SCHEMA,
        method="append_adapt",
        derivation_policy=APPEND_INTEGRITY_DERIVATION,
        reporting_only=True,
        controller_decision_influence=False,
        finite_values_passed=finite_passed,
        checked_energy_value_count=energy_count,
        checked_parameter_value_count=parameter_count,
        nonfinite_value_paths=tuple(nonfinite),
        sector_diagnostic_policy=str(diagnostic["policy"]),
        state_fingerprint=projective_state_fingerprint(
            np.asarray(final_state, dtype=complex).reshape(-1)
        ),
        sector_leak_threshold=float(
            diagnostic["sector_leak_threshold"]
        ),
        fixed_count_sector_probability=float(
            diagnostic["fixed_count_sector_probability"]
        ),
        fixed_count_sector_leak_probability=float(
            diagnostic["fixed_count_sector_leak_probability"]
        ),
        sector_leak_flag=sector_leak_flag,
        boson_legal_probability_min=diagnostic[
            "boson_legal_probability_min"
        ],
        boson_illegal_probability_max=diagnostic[
            "boson_illegal_probability_max"
        ],
        boson_truncation_leak_flag=boson_leak_flag,
        accepted_energy_transitions=tuple(transitions),
        accepted_energy_integrity_passed=energy_integrity,
        integrity_passed=integrity_passed,
    )


__all__ = [
    "APPEND_INTEGRITY_DERIVATION",
    "APPEND_NONWORSENING_ABSOLUTE_TOLERANCE",
    "AcceptedEnergyTransitionIntegrityReceipt",
    "ENERGY_TRANSITION_INTEGRITY_SCHEMA",
    "NUMERICAL_PHYSICAL_INTEGRITY_SCHEMA",
    "NumericalPhysicalIntegrityReceipt",
    "RA_INTEGRITY_DERIVATION",
    "SECTOR_DIAGNOSTIC_POLICY",
    "SECTOR_LEAK_THRESHOLD",
    "build_append_numerical_physical_integrity",
    "build_ra_numerical_physical_integrity",
    "numerical_physical_integrity_from_mapping",
    "sector_probability",
]
