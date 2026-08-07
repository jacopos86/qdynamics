"""Canonical Route-A shortlist population helpers.

Phases 0-2 cap macro-operator identities, while Phase III caps unique Pauli
children.  Candidate insertion positions remain separate records after an
identity survives a shortlist.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

from pipelines.scaffold.hh_continuation_generators import (
    serialize_polynomial_terms_exyz,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


ROUTE_A_SHORTLIST_UNIT_MACRO_OPERATOR = "macro_operator_identity"
ROUTE_A_SHORTLIST_UNIT_PAULI_CHILD = "pauli_child_identity"
CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1 = "global_pauli_word_v1"
CHILD_IDENTITY_POLICY_PARENT_QUALIFIED_LEGACY_V1 = "parent_qualified_legacy_v1"
PAULI_CHILD_IDENTITY_NORMALIZATION_PROJECTIVE_V1 = (
    "projective_normalized_pauli_polynomial_v1"
)
CHILD_IDENTITY_POLICIES = frozenset(
    {
        CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
        CHILD_IDENTITY_POLICY_PARENT_QUALIFIED_LEGACY_V1,
    }
)


def _finite_score(record: Mapping[str, Any], key: str | None) -> float:
    if key in {None, ""}:
        return float("-inf")
    try:
        value = float(record.get(str(key), float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")
    return value if math.isfinite(value) else float("-inf")


def _record_rank_key(
    record: Mapping[str, Any],
    *,
    score_key: str,
    tie_break_score_key: str | None,
) -> tuple[float, float, int, str]:
    return (
        -_finite_score(record, score_key),
        -_finite_score(record, tie_break_score_key),
        int(record.get("position_id", -1)),
        str(record.get("candidate_label", "")),
    )


def macro_operator_identity(record: Mapping[str, Any]) -> str:
    """Return a stable macro identity shared by all insertion positions."""

    pool_index = record.get("candidate_pool_index")
    try:
        pool_index_value = int(pool_index)
    except (TypeError, ValueError):
        pool_index_value = -1
    if pool_index_value >= 0:
        return f"pool:{pool_index_value}"

    feature = record.get("feature")
    metadata = getattr(feature, "generator_metadata", None)
    if not isinstance(metadata, Mapping) and isinstance(record.get("generator_metadata"), Mapping):
        metadata = record.get("generator_metadata")
    if isinstance(metadata, Mapping) and metadata.get("generator_id") not in {None, ""}:
        return f"generator:{metadata['generator_id']}"

    label = str(record.get("candidate_label") or getattr(record.get("candidate_term"), "label", ""))
    return f"label:{label}"


def _canonical_polynomial_direction_payload(
    candidate_term: Any,
    *,
    tol: float = 1e-12,
) -> list[dict[str, Any]] | None:
    """Serialize one Pauli-polynomial direction modulo a global scalar."""

    polynomial = getattr(candidate_term, "polynomial", None)
    if polynomial is None:
        return None
    try:
        terms = serialize_polynomial_terms_exyz(polynomial)
    except Exception:
        return None
    combined: dict[tuple[int, str], complex] = defaultdict(complex)
    for row in terms:
        key = (
            int(row.get("nq", len(str(row.get("pauli_exyz", ""))))),
            str(row.get("pauli_exyz", "")).lower(),
        )
        combined[key] += complex(
            float(row.get("coeff_re", 0.0)),
            float(row.get("coeff_im", 0.0)),
        )
    retained = [
        (key, coeff)
        for key, coeff in sorted(combined.items())
        if abs(coeff) > float(tol)
    ]
    if not retained:
        return None
    norm = math.sqrt(sum(abs(coeff) ** 2 for _, coeff in retained))
    if norm <= float(tol):
        return None
    anchor = retained[0][1]
    global_phase = anchor / abs(anchor)
    decimals = max(0, min(15, int(round(-math.log10(float(tol))))))

    def _rounded(value: float) -> float:
        rounded = round(float(value), decimals)
        return 0.0 if abs(rounded) <= float(tol) else float(rounded)

    return [
        {
            "nq": int(key[0]),
            "pauli_exyz": str(key[1]),
            "coeff_re": _rounded((coeff / (norm * global_phase)).real),
            "coeff_im": _rounded((coeff / (norm * global_phase)).imag),
        }
        for key, coeff in retained
    ]


def _polynomial_digest(candidate_term: Any) -> str | None:
    terms = _canonical_polynomial_direction_payload(candidate_term)
    if terms is None:
        return None
    payload = json.dumps(
        terms,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pauli_child_identity(record: Mapping[str, Any]) -> str:
    """Identify a normalized Pauli direction independently of parent/position."""

    digest = _polynomial_digest(record.get("candidate_term"))
    if digest is not None:
        return f"pauli:{digest}"

    feature = record.get("feature")
    child_labels = getattr(feature, "runtime_split_child_labels", None)
    if isinstance(child_labels, Sequence) and not isinstance(child_labels, (str, bytes)):
        normalized = tuple(str(value) for value in child_labels)
        if normalized:
            return "child_labels:" + "|".join(normalized)

    label = str(record.get("candidate_label") or getattr(record.get("candidate_term"), "label", ""))
    return f"label:{label}"


def canonicalize_pauli_child_direction(
    candidate_term: Any,
    *,
    tol: float = 1e-12,
) -> tuple[Any, dict[str, Any]]:
    """Return the deterministic unit-norm representative of one direction."""

    normalized_terms = _canonical_polynomial_direction_payload(
        candidate_term,
        tol=float(tol),
    )
    polynomial = getattr(candidate_term, "polynomial", None)
    if normalized_terms is None or polynomial is None:
        return candidate_term, {
            "schema": "route_a_child_direction_normalization_v1",
            "status": "unavailable",
            "reason": "candidate_polynomial_direction_unavailable",
        }

    source_terms = serialize_polynomial_terms_exyz(polynomial)
    source_coefficients: dict[tuple[int, str], complex] = defaultdict(complex)
    for row in source_terms:
        key = (
            int(row.get("nq", len(str(row.get("pauli_exyz", ""))))),
            str(row.get("pauli_exyz", "")).lower(),
        )
        source_coefficients[key] += complex(
            float(row.get("coeff_re", 0.0)),
            float(row.get("coeff_im", 0.0)),
        )
    retained_source = [
        (key, coefficient)
        for key, coefficient in sorted(source_coefficients.items())
        if abs(coefficient) > float(tol)
    ]
    source_norm = math.sqrt(
        sum(abs(coefficient) ** 2 for _, coefficient in retained_source)
    )
    source_anchor = retained_source[0][1]
    source_phase = source_anchor / abs(source_anchor)

    canonical_polynomial = PauliPolynomial(
        "JW",
        [
            PauliTerm(
                int(row["nq"]),
                ps=str(row["pauli_exyz"]),
                pc=complex(
                    float(row.get("coeff_re", 0.0)),
                    float(row.get("coeff_im", 0.0)),
                ),
            )
            for row in normalized_terms
        ],
    )
    canonical_term = AnsatzTerm(
        label=str(getattr(candidate_term, "label", "")),
        polynomial=canonical_polynomial,
        execution_mode=str(
            getattr(candidate_term, "execution_mode", "termwise_product")
        ),
    )
    return canonical_term, {
        "schema": "route_a_child_direction_normalization_v1",
        "status": "normalized",
        "source_coefficient_l2_norm": float(source_norm),
        "source_anchor_phase_re": float(source_phase.real),
        "source_anchor_phase_im": float(source_phase.imag),
        "canonical_coefficient_l2_norm": 1.0,
        "canonical_anchor_positive_real": True,
        "global_scalar_sign_phase_removed": True,
        "term_count": int(len(normalized_terms)),
    }


def child_identity_for_policy(
    record: Mapping[str, Any],
    *,
    policy: str,
) -> str:
    policy_key = str(policy).strip().lower()
    if policy_key not in CHILD_IDENTITY_POLICIES:
        raise ValueError(
            f"child identity policy must be one of {sorted(CHILD_IDENTITY_POLICIES)}; "
            f"got {policy!r}."
        )
    child_identity = pauli_child_identity(record)
    if policy_key == CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1:
        return child_identity
    parent_label = str(
        record.get("runtime_split_parent_label")
        or getattr(record.get("feature"), "runtime_split_parent_label", "")
        or ""
    )
    return f"parent:{parent_label}|{child_identity}"


@dataclass(frozen=True)
class IdentityPopulation:
    representatives: tuple[dict[str, Any], ...]
    records_by_identity: Mapping[str, tuple[dict[str, Any], ...]]

    @property
    def identity_count(self) -> int:
        return int(len(self.representatives))

    @property
    def record_count(self) -> int:
        return int(sum(len(rows) for rows in self.records_by_identity.values()))


def identity_population(
    records: Sequence[Mapping[str, Any]],
    *,
    identity_key: Callable[[Mapping[str, Any]], str],
    score_key: str,
    tie_break_score_key: str | None = None,
) -> IdentityPopulation:
    """Collapse records to the best-position representative per identity."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        copied = dict(record)
        grouped[str(identity_key(copied))].append(copied)

    representatives: list[dict[str, Any]] = []
    normalized_groups: dict[str, tuple[dict[str, Any], ...]] = {}
    for identity, rows in grouped.items():
        ranked = sorted(
            rows,
            key=lambda row: _record_rank_key(
                row,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            ),
        )
        representative = dict(ranked[0])
        representative["route_a_shortlist_identity"] = str(identity)
        representative["route_a_identity_position_count"] = int(len(ranked))
        representatives.append(representative)
        normalized_groups[str(identity)] = tuple(dict(row) for row in ranked)

    representatives.sort(
        key=lambda row: _record_rank_key(
            row,
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        )
    )
    return IdentityPopulation(
        representatives=tuple(representatives),
        records_by_identity=normalized_groups,
    )


def expand_selected_identities(
    population: IdentityPopulation,
    selected_representatives: Sequence[Mapping[str, Any]],
    *,
    shortlist_flag: str | None,
    shortlist_unit: str,
    feature_updater: Callable[[Any, Mapping[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Restore every position record for each selected identity."""

    expanded: list[dict[str, Any]] = []
    selected_count = int(len(selected_representatives))
    for identity_rank, representative in enumerate(selected_representatives, start=1):
        identity = str(representative.get("route_a_shortlist_identity", ""))
        rows = population.records_by_identity.get(identity, ())
        for position_rank, source in enumerate(rows, start=1):
            row = dict(source)
            updates = {
                "route_a_shortlist_unit": str(shortlist_unit),
                "route_a_shortlist_identity": str(identity),
                "route_a_identity_rank": int(identity_rank),
                "route_a_identity_shortlist_size": int(selected_count),
                "route_a_identity_position_rank": int(position_rank),
                "route_a_identity_position_count": int(len(rows)),
            }
            if shortlist_flag is not None:
                updates[str(shortlist_flag)] = True
            row.update(updates)
            feature = row.get("feature")
            if feature_updater is not None and feature is not None:
                row["feature"] = feature_updater(feature, updates)
            expanded.append(row)
    return expanded


def deduplicate_child_position_records(
    records: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    tie_break_score_key: str | None = None,
    identity_policy: str = CHILD_IDENTITY_POLICY_PARENT_QUALIFIED_LEGACY_V1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate by the configured child identity and insertion position.

    The same Pauli word at different positions remains as alternative records.
    Under the global policy, parent labels are retained only as provenance.
    """

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        copied = dict(record)
        key = (
            child_identity_for_policy(copied, policy=identity_policy),
            int(copied.get("position_id", -1)),
        )
        grouped[key].append(copied)

    deduplicated: list[dict[str, Any]] = []
    duplicate_count = 0
    for (identity, position), rows in grouped.items():
        ranked = sorted(
            rows,
            key=lambda row: _record_rank_key(
                row,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            ),
        )
        chosen = dict(ranked[0])
        if str(identity_policy) == CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1:
            canonical_term, normalization = canonicalize_pauli_child_direction(
                chosen.get("candidate_term")
            )
            chosen["candidate_term"] = canonical_term
            chosen["route_a_child_direction_normalization"] = normalization
        parent_label_set: set[str] = set()
        for row in rows:
            existing = row.get("route_a_child_parent_labels")
            if isinstance(existing, Sequence) and not isinstance(
                existing, (str, bytes)
            ):
                parent_label_set.update(str(value) for value in existing if str(value))
            direct = str(
                row.get("runtime_split_parent_label")
                or getattr(row.get("feature"), "runtime_split_parent_label", "")
                or ""
            )
            if direct:
                parent_label_set.add(direct)
        parent_labels = sorted(parent_label_set)
        chosen.update(
            {
                "route_a_child_identity": str(identity),
                "route_a_global_pauli_identity": pauli_child_identity(chosen),
                "route_a_child_position": int(position),
                "route_a_child_parent_labels": parent_labels,
                "route_a_child_parent_count": int(len(parent_labels)),
            }
        )
        deduplicated.append(chosen)
        duplicate_count += int(max(0, len(rows) - 1))

    deduplicated.sort(
        key=lambda row: _record_rank_key(
            row,
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        )
    )
    identity_count = int(
        len(
            {
                child_identity_for_policy(row, policy=identity_policy)
                for row in deduplicated
            }
        )
    )
    return deduplicated, {
        "schema": "route_a_global_child_population_v1",
        "input_record_count": int(len(records)),
        "deduplicated_record_count": int(len(deduplicated)),
        "duplicate_record_count": int(duplicate_count),
        "unique_child_identity_count": int(identity_count),
        "position_alternatives_preserved": True,
        "deduplication_key": (
            f"{str(identity_policy)}_child_identity_plus_position"
        ),
        "child_identity_policy": str(identity_policy),
        "identity_normalization": (
            PAULI_CHILD_IDENTITY_NORMALIZATION_PROJECTIVE_V1
        ),
        "global_scalar_sign_phase_invariant": True,
        "canonical_direction_representative": (
            "unit_l2_norm_positive_anchor_v1"
            if str(identity_policy) == CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1
            else "source_record_legacy"
        ),
        "relative_coefficients_preserved": True,
    }


__all__ = [
    "IdentityPopulation",
    "CHILD_IDENTITY_POLICIES",
    "CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1",
    "CHILD_IDENTITY_POLICY_PARENT_QUALIFIED_LEGACY_V1",
    "PAULI_CHILD_IDENTITY_NORMALIZATION_PROJECTIVE_V1",
    "ROUTE_A_SHORTLIST_UNIT_MACRO_OPERATOR",
    "ROUTE_A_SHORTLIST_UNIT_PAULI_CHILD",
    "deduplicate_child_position_records",
    "child_identity_for_policy",
    "expand_selected_identities",
    "identity_population",
    "macro_operator_identity",
    "pauli_child_identity",
    "canonicalize_pauli_child_direction",
]
