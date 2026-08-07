"""Binary-padding eligibility for canonical Route-A Pauli children."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.static_adapt.builders.legal_subspace_filter import (
    boson_legal_register_indices,
    legal_subspace_basis_for_problem,
    pauli_action_on_basis_index,
    pauli_word_illegal_hit_count,
)
from pipelines.static_adapt.route_a_shortlists import pauli_child_identity
from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


ROUTE_A_CHILD_PADDING_HARD_FILTER_V1 = "legal_codeword_hard_filter_v1"
ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1 = "exact_projected_grouped_v1"
ROUTE_A_CHILD_PADDING_HARD_FILTER_NPH2_V1 = (
    "nph2_legal_codeword_hard_filter_v1"
)
ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1 = (
    "nph2_exact_projected_grouped_v1"
)
ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1 = (
    "unchecked_diagnostic_v1"
)
ROUTE_A_CHILD_PADDING_POLICIES = frozenset(
    {
        ROUTE_A_CHILD_PADDING_HARD_FILTER_V1,
        ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
        ROUTE_A_CHILD_PADDING_HARD_FILTER_NPH2_V1,
        ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1,
        ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1,
    }
)
ROUTE_A_CHILD_PADDING_HARD_FILTER_POLICIES = frozenset(
    {
        ROUTE_A_CHILD_PADDING_HARD_FILTER_V1,
        ROUTE_A_CHILD_PADDING_HARD_FILTER_NPH2_V1,
    }
)
ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_POLICIES = frozenset(
    {
        ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
        ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1,
    }
)
ROUTE_A_CHILD_PADDING_SCHEMA = "route_a_child_padding_filter_v1"
ROUTE_A_CHILD_PADDING_PROJECTION_SCHEMA = (
    "route_a_child_padding_exact_projection_v1"
)
_PROJECTION_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RouteAChildPaddingConfig:
    policy: str = ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1
    problem_key: str | None = None
    num_sites: int | None = None
    n_ph_max: int | None = None
    boson_encoding: str | None = None
    total_register_width: int | None = None

    def __post_init__(self) -> None:
        policy = str(self.policy)
        if policy not in ROUTE_A_CHILD_PADDING_POLICIES:
            raise ValueError(
                "Route-A child-padding policy must be one of "
                f"{sorted(ROUTE_A_CHILD_PADDING_POLICIES)}."
            )
        active_policies = (
            ROUTE_A_CHILD_PADDING_HARD_FILTER_POLICIES
            | ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_POLICIES
        )
        if policy not in active_policies:
            return
        if str(self.problem_key).strip().lower() != "hh":
            raise ValueError(
                f"{policy} currently requires problem_key='hh'."
            )
        if str(self.boson_encoding).strip().lower() != "binary":
            raise ValueError(
                f"{policy} requires binary encoding."
            )
        n_ph_max = int(self.n_ph_max if self.n_ph_max is not None else -1)
        if n_ph_max < 1:
            raise ValueError(
                f"{policy} requires n_ph_max >= 1."
            )
        if policy in {
            ROUTE_A_CHILD_PADDING_HARD_FILTER_NPH2_V1,
            ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1,
        } and n_ph_max != 2:
            raise ValueError(f"{policy} requires n_ph_max=2.")
        for name in ("num_sites", "total_register_width"):
            value = getattr(self, name)
            if value is None or int(value) < 1:
                raise ValueError(f"{name} must be positive for the hard filter.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": str(self.policy),
            "problem_key": self.problem_key,
            "num_sites": self.num_sites,
            "n_ph_max": self.n_ph_max,
            "boson_encoding": self.boson_encoding,
            "total_register_width": self.total_register_width,
        }


_ONE_QUBIT_PAULI_MATRICES = {
    "e": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


@lru_cache(maxsize=None)
def _pauli_matrix_exyz(label: str) -> np.ndarray:
    matrix = np.asarray([[1.0 + 0.0j]])
    for symbol in str(label):
        try:
            local = _ONE_QUBIT_PAULI_MATRICES[str(symbol)]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Pauli symbol {symbol!r} in {label!r}."
            ) from exc
        matrix = np.kron(matrix, local)
    return np.asarray(matrix, dtype=complex)


@lru_cache(maxsize=None)
def _projected_local_pauli_components(
    label: str,
    *,
    n_ph_max: int,
    boson_encoding: str,
) -> tuple[tuple[str, complex], ...]:
    """Return a minimal Pauli extension of P_legal P P_legal on one site."""

    local_label = str(label)
    qpb = int(
        boson_qubits_per_site(
            int(n_ph_max),
            str(boson_encoding),
        )
    )
    if len(local_label) != qpb:
        raise ValueError(
            f"Local boson Pauli width {len(local_label)} does not match qpb={qpb}."
        )
    if all(symbol in {"e", "z"} for symbol in local_label):
        return ((local_label, 1.0 + 0.0j),)

    legal_indices = boson_legal_register_indices(
        num_sites=1,
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
    )
    dim = 1 << qpb
    projector = np.zeros((dim, dim), dtype=complex)
    projector[list(legal_indices), list(legal_indices)] = 1.0
    projected = (
        projector @ _pauli_matrix_exyz(local_label) @ projector
    )
    components: list[tuple[str, complex]] = []
    for symbols in itertools.product("exyz", repeat=qpb):
        basis_label = "".join(symbols)
        basis_matrix = _pauli_matrix_exyz(basis_label)
        coefficient = np.trace(basis_matrix.conj().T @ projected) / float(dim)
        if abs(complex(coefficient)) <= _PROJECTION_TOLERANCE:
            continue
        if abs(float(complex(coefficient).imag)) > _PROJECTION_TOLERANCE:
            raise ValueError(
                "Projected Hermitian child acquired a non-real Pauli coefficient: "
                f"{basis_label}={coefficient}."
            )
        components.append(
            (basis_label, complex(float(complex(coefficient).real), 0.0))
        )
    return tuple(components)


def project_route_a_child_polynomial(
    polynomial: PauliPolynomial,
    *,
    config: RouteAChildPaddingConfig,
) -> tuple[PauliPolynomial | None, dict[str, Any]]:
    """Project a raw Pauli child into an exact legal grouped generator."""

    if str(config.policy) not in ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_POLICIES:
        return polynomial, {
            "schema": ROUTE_A_CHILD_PADDING_PROJECTION_SCHEMA,
            "active": False,
            "policy": str(config.policy),
            "reason": "projection_policy_inactive",
        }

    total_width = int(config.total_register_width)
    non_boson_width = 2 * int(config.num_sites)
    boson_width = int(total_width - non_boson_width)
    qpb = int(
        boson_qubits_per_site(
            int(config.n_ph_max),
            str(config.boson_encoding),
        )
    )
    if boson_width != int(config.num_sites) * qpb:
        raise ValueError(
            "Route-A child projection register layout is inconsistent: "
            f"boson_width={boson_width}, num_sites={config.num_sites}, qpb={qpb}."
        )

    coefficients: dict[str, complex] = {}
    raw_term_count = 0
    affected_site_blocks = 0
    for term in polynomial.return_polynomial():
        raw_coefficient = complex(term.p_coeff)
        if abs(raw_coefficient) <= _PROJECTION_TOLERANCE:
            continue
        if abs(float(raw_coefficient.imag)) > _PROJECTION_TOLERANCE:
            raise ValueError(
                "Route-A projected child requires a Hermitian raw Pauli direction; "
                f"got coefficient {raw_coefficient}."
            )
        label = str(term.pw2strng())
        if len(label) != total_width:
            raise ValueError(
                "Route-A child projection register width mismatch: "
                f"{len(label)}!={total_width}."
            )
        raw_term_count += 1
        boson_label = label[:boson_width]
        non_boson_label = label[boson_width:]
        block_components: list[tuple[tuple[str, complex], ...]] = []
        for offset in range(0, boson_width, qpb):
            block = boson_label[offset : offset + qpb]
            if any(symbol in {"x", "y"} for symbol in block):
                affected_site_blocks += 1
            block_components.append(
                _projected_local_pauli_components(
                    block,
                    n_ph_max=int(config.n_ph_max),
                    boson_encoding=str(config.boson_encoding),
                )
            )
        for component_choice in itertools.product(*block_components):
            projected_boson_label = "".join(
                component[0] for component in component_choice
            )
            projected_coefficient = raw_coefficient
            for _component_label, component_coefficient in component_choice:
                projected_coefficient *= complex(component_coefficient)
            projected_label = projected_boson_label + non_boson_label
            coefficients[projected_label] = (
                coefficients.get(projected_label, 0.0 + 0.0j)
                + projected_coefficient
            )

    terms: list[PauliTerm] = []
    for label in sorted(coefficients):
        coefficient = complex(coefficients[label])
        if abs(coefficient) <= _PROJECTION_TOLERANCE:
            continue
        if abs(float(coefficient.imag)) > _PROJECTION_TOLERANCE:
            raise ValueError(
                "Projected Route-A child is not Hermitian within tolerance: "
                f"{label}={coefficient}."
            )
        terms.append(
            PauliTerm(
                total_width,
                ps=str(label),
                pc=float(coefficient.real),
            )
        )
    if not terms:
        return None, {
            "schema": ROUTE_A_CHILD_PADDING_PROJECTION_SCHEMA,
            "active": True,
            "policy": str(config.policy),
            "reason": "projection_is_zero",
            "raw_term_count": int(raw_term_count),
            "projected_term_count": 0,
        }
    projected_polynomial = PauliPolynomial("JW", terms)
    return projected_polynomial, {
        "schema": ROUTE_A_CHILD_PADDING_PROJECTION_SCHEMA,
        "active": True,
        "policy": str(config.policy),
        "reason": "exact_local_block_projection_applied",
        "projection_equivalence": (
            "P_legal_P_P_legal_on_legal_subspace_minimal_block_extension_v1"
        ),
        "raw_term_count": int(raw_term_count),
        "projected_term_count": int(
            projected_polynomial.count_number_terms()
        ),
        "affected_site_block_count": int(affected_site_blocks),
        "recommended_execution_mode": "grouped_exact",
        "applied_before_child_phase1_evaluation": True,
        "applied_before_global_projected_identity_deduplication": True,
    }


def _grouped_polynomial_legality_stats(
    polynomial: Any,
    *,
    legal_indices: Sequence[int],
    legal_set: set[int],
) -> dict[str, Any]:
    coefficients: dict[str, complex] = {}
    for term in polynomial.return_polynomial():
        label = str(term.pw2strng())
        coefficients[label] = coefficients.get(label, 0.0 + 0.0j) + complex(
            term.p_coeff
        )
    illegal_basis_hit_count = 0
    max_illegal_action_norm = 0.0
    for basis_index in legal_indices:
        amplitudes: dict[int, complex] = {}
        for label, coefficient in coefficients.items():
            if abs(coefficient) <= _PROJECTION_TOLERANCE:
                continue
            out_index, phase = pauli_action_on_basis_index(label, int(basis_index))
            amplitudes[out_index] = (
                amplitudes.get(out_index, 0.0 + 0.0j)
                + coefficient * phase
            )
        illegal_norm = float(
            sum(
                abs(amplitude) ** 2
                for out_index, amplitude in amplitudes.items()
                if int(out_index) not in legal_set
            )
            ** 0.5
        )
        if illegal_norm > _PROJECTION_TOLERANCE:
            illegal_basis_hit_count += 1
            max_illegal_action_norm = max(
                max_illegal_action_norm,
                illegal_norm,
            )
    return {
        "legal_preserving": bool(illegal_basis_hit_count == 0),
        "illegal_basis_hit_count": int(illegal_basis_hit_count),
        "max_illegal_action_norm": float(max_illegal_action_norm),
    }


def _singleton_pauli_label(
    record: Mapping[str, Any],
    *,
    coefficient_tolerance: float = 1e-12,
) -> tuple[str | None, str | None]:
    candidate_term = record.get("candidate_term")
    polynomial = getattr(candidate_term, "polynomial", None)
    if polynomial is None or not hasattr(polynomial, "return_polynomial"):
        return None, "candidate_polynomial_missing"
    coefficients: dict[str, complex] = {}
    for term in polynomial.return_polynomial():
        label = str(term.pw2strng())
        coefficients[label] = coefficients.get(label, 0.0 + 0.0j) + complex(
            term.p_coeff
        )
    labels = sorted(
        label
        for label, coefficient in coefficients.items()
        if abs(complex(coefficient)) > float(coefficient_tolerance)
    )
    if len(labels) != 1:
        return None, "canonical_child_is_not_single_pauli_word"
    return str(labels[0]), None


def filter_route_a_child_padding_records(
    records: Sequence[Mapping[str, Any]],
    *,
    config: RouteAChildPaddingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enforce or validate the configured binary child-padding policy."""

    input_records = [dict(record) for record in records]
    if str(config.policy) == ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1:
        return input_records, {
            "schema": ROUTE_A_CHILD_PADDING_SCHEMA,
            "policy_requested": str(config.policy),
            "policy_effective": str(config.policy),
            "active": False,
            "reason": "diagnostic_compatibility_mode",
            "applied_after_global_child_deduplication": True,
            "applied_before_child_phase1": True,
            "input_record_count": int(len(input_records)),
            "remaining_record_count": int(len(input_records)),
            "rejected_record_count": 0,
            "rejected_identity_count": 0,
            "rejected_identities": [],
            "rejected_records": [],
        }

    layout = legal_subspace_basis_for_problem(
        problem_key=str(config.problem_key),
        num_sites=int(config.num_sites),
        n_ph_max=int(config.n_ph_max),
        boson_encoding=str(config.boson_encoding),
        total_register_width=int(config.total_register_width),
    )
    legal_indices = tuple(int(value) for value in layout["legal_indices"])
    legal_set = set(legal_indices)
    label_illegal_hits: dict[str, int] = {}
    grouped_stats_by_identity: dict[str, dict[str, Any]] = {}
    retained: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    rejected_identities: dict[str, dict[str, Any]] = {}
    for record in input_records:
        identity = str(pauli_child_identity(record))
        label: str | None = None
        illegal_hit_count = 0
        max_illegal_action_norm = 0.0
        if (
            str(config.policy)
            in ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_POLICIES
        ):
            candidate_term = record.get("candidate_term")
            polynomial = getattr(candidate_term, "polynomial", None)
            if polynomial is None or not hasattr(polynomial, "return_polynomial"):
                rejection_reason = "candidate_polynomial_missing"
            else:
                if identity not in grouped_stats_by_identity:
                    grouped_stats_by_identity[identity] = (
                        _grouped_polynomial_legality_stats(
                            polynomial,
                            legal_indices=legal_indices,
                            legal_set=legal_set,
                        )
                    )
                grouped_stats = grouped_stats_by_identity[identity]
                illegal_hit_count = int(
                    grouped_stats["illegal_basis_hit_count"]
                )
                max_illegal_action_norm = float(
                    grouped_stats["max_illegal_action_norm"]
                )
                rejection_reason = (
                    None
                    if bool(grouped_stats["legal_preserving"])
                    else "projected_grouped_child_is_not_legal_preserving"
                )
        else:
            label, extraction_reason = _singleton_pauli_label(record)
            rejection_reason = extraction_reason
            if label is not None and len(label) != int(config.total_register_width):
                rejection_reason = "candidate_register_width_mismatch"
            elif label is not None:
                if label not in label_illegal_hits:
                    label_illegal_hits[label] = pauli_word_illegal_hit_count(
                        label,
                        legal_indices=legal_indices,
                        legal_set=legal_set,
                    )
                illegal_hit_count = int(label_illegal_hits[label])
                if illegal_hit_count > 0:
                    rejection_reason = (
                        "pauli_word_maps_legal_codeword_to_padding"
                    )
        if rejection_reason is None:
            retained.append(record)
            continue

        rejected = {
            "identity": identity,
            "pauli_label_exyz": label,
            "candidate_label": str(record.get("candidate_label", "")),
            "position_id": int(record.get("position_id", -1)),
            "parent_labels": [
                str(value)
                for value in record.get("route_a_child_parent_labels", [])
            ],
            "reason": str(rejection_reason),
            "illegal_basis_hit_count": int(illegal_hit_count),
            "max_illegal_action_norm": float(max_illegal_action_norm),
        }
        rejected_records.append(rejected)
        identity_payload = rejected_identities.setdefault(
            identity,
            {
                "identity": identity,
                "pauli_label_exyz": label,
                "reason": str(rejection_reason),
                "illegal_basis_hit_count": int(illegal_hit_count),
                "max_illegal_action_norm": float(max_illegal_action_norm),
                "parent_labels": set(),
                "position_ids": set(),
                "record_count": 0,
            },
        )
        identity_payload["parent_labels"].update(rejected["parent_labels"])
        identity_payload["position_ids"].add(int(rejected["position_id"]))
        identity_payload["record_count"] = int(identity_payload["record_count"]) + 1

    rejected_identity_rows = []
    for identity in sorted(rejected_identities):
        payload = dict(rejected_identities[identity])
        payload["parent_labels"] = sorted(payload["parent_labels"])
        payload["position_ids"] = sorted(payload["position_ids"])
        rejected_identity_rows.append(payload)

    projected_policy = bool(
        str(config.policy) in ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_POLICIES
    )
    rejected_record_sample_limit = 20

    return retained, {
        "schema": ROUTE_A_CHILD_PADDING_SCHEMA,
        "policy_requested": str(config.policy),
        "policy_effective": str(config.policy),
        "active": True,
        "reason": (
            "projected_grouped_legality_validated"
            if projected_policy
            else "hard_filter_applied"
        ),
        "applied_after_global_child_deduplication": True,
        "applied_before_child_phase1": True,
        "projection_applied_before_child_phase1_evaluation": projected_policy,
        "projection_applied_before_global_child_deduplication": projected_policy,
        "validation_mode": (
            "grouped_polynomial_legal_action_v1"
            if projected_policy
            else "singleton_pauli_global_codeword_action_v1"
        ),
        "input_record_count": int(len(input_records)),
        "remaining_record_count": int(len(retained)),
        "rejected_record_count": int(len(rejected_records)),
        "rejected_identity_count": int(len(rejected_identities)),
        "rejected_identities": rejected_identity_rows,
        "rejected_records": rejected_records[:rejected_record_sample_limit],
        "rejected_records_truncated": bool(
            len(rejected_records) > rejected_record_sample_limit
        ),
        "rejected_record_sample_limit": int(rejected_record_sample_limit),
        "legal_subspace": {
            key: value
            for key, value in layout.items()
            if key != "legal_indices"
        },
    }


__all__ = [
    "ROUTE_A_CHILD_PADDING_HARD_FILTER_POLICIES",
    "ROUTE_A_CHILD_PADDING_HARD_FILTER_V1",
    "ROUTE_A_CHILD_PADDING_HARD_FILTER_NPH2_V1",
    "ROUTE_A_CHILD_PADDING_POLICIES",
    "ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_POLICIES",
    "ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1",
    "ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_NPH2_V1",
    "ROUTE_A_CHILD_PADDING_PROJECTION_SCHEMA",
    "ROUTE_A_CHILD_PADDING_SCHEMA",
    "ROUTE_A_CHILD_PADDING_UNCHECKED_DIAGNOSTIC_V1",
    "RouteAChildPaddingConfig",
    "filter_route_a_child_padding_records",
    "project_route_a_child_polynomial",
]
