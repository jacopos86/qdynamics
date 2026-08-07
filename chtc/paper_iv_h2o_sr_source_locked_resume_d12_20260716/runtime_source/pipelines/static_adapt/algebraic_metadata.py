#!/usr/bin/env python3
"""Exact algebraic metadata helpers for static ADAPT staged selection.

This module is intentionally pure helper code for Slice A of the static ADAPT
algebraic-lane update.  It does not wire any live pipeline behavior.  Repo-native
algebraic lanes require exact Pauli expansions; missing exact expansions are
reported as bugs instead of silently falling back to label/support metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

LANE_FLAT = "flat"
LANE_CURV = "curv"
LANE_DISJ = "disj"
LANE_MIX = "mix"
LANES_PHASE1 = (LANE_FLAT, LANE_CURV, LANE_DISJ, LANE_MIX)

EXACTNESS_EXACT = "exact"
EXACTNESS_APPROX = "approx"

RELATION_FLAT_COMM = "flat_comm"
RELATION_CURV_NONCOMM = "curv_noncomm"
RELATION_DISJ_COMM = "disj_comm"
RELATION_APPROX_OR_UNKNOWN = "approx_or_unknown"


class AlgebraicMetadataError(ValueError):
    """Raised when exact algebraic metadata cannot be built safely."""


@dataclass(frozen=True)
class SerializedPauliExpansionTerm:
    """One exact serialized Pauli word in repo-native e/x/y/z order."""

    pauli_exyz: str
    coeff_re: float
    coeff_im: float
    nq: int

    @property
    def coeff(self) -> complex:
        return complex(float(self.coeff_re), float(self.coeff_im))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SerializedPauliExpansionTerm":
        nq = int(raw.get("nq", 0))
        word = normalize_pauli_word_exyz(raw.get("pauli_exyz", ""))
        if nq <= 0:
            raise AlgebraicMetadataError(f"Serialized Pauli term has invalid nq={nq!r}.")
        if len(word) != nq:
            raise AlgebraicMetadataError(
                f"Serialized Pauli word length {len(word)} does not match nq={nq}: {word!r}."
            )
        if "coeff_re" not in raw or "coeff_im" not in raw:
            raise AlgebraicMetadataError(
                f"Serialized Pauli term {word!r} is missing coeff_re/coeff_im fields."
            )
        return cls(
            pauli_exyz=word,
            coeff_re=float(raw["coeff_re"]),
            coeff_im=float(raw["coeff_im"]),
            nq=int(nq),
        )


@dataclass(frozen=True)
class GeneratorAlgebraicExpansion:
    """Exact or explicitly approximate expansion metadata for one generator."""

    key: str
    label: str
    generator_id: str | None
    terms: tuple[SerializedPauliExpansionTerm, ...]
    support_qubits: tuple[int, ...]
    exactness: str = EXACTNESS_EXACT
    source: str = "registry_serialized_terms"


@dataclass(frozen=True)
class AlgebraicPairMetadata:
    """Pair relation used for local lane assignment and prune compensator pools."""

    lhs_key: str
    rhs_key: str
    lhs_label: str
    rhs_label: str
    support_overlap: bool
    commutes: bool | None
    exactness: str
    relation: str
    support_overlap_qubits: tuple[int, ...] = ()
    commutator_l1_norm: float | None = None


@dataclass(frozen=True)
class AlgebraicLocalContextSummary:
    """Local algebraic context counts for a candidate against scaffold records."""

    candidate_key: str
    candidate_label: str
    context_keys: tuple[str, ...]
    context_labels: tuple[str, ...]
    n_flat: int
    n_curv: int
    n_disj: int
    n_approx: int
    lane: str
    quality: str


@dataclass
class AlgebraicMetadataIndex:
    """Lazy pair-metadata index keyed by exact generator expansion keys."""

    expansions_by_key: Mapping[str, GeneratorAlgebraicExpansion]
    label_to_keys: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    pair_cache: dict[tuple[str, str, float], AlgebraicPairMetadata] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expansions = {str(key): val for key, val in self.expansions_by_key.items()}
        labels: dict[str, list[str]] = {}
        for key, expansion in expansions.items():
            labels.setdefault(str(expansion.label), []).append(str(key))
        for label, keys in self.label_to_keys.items():
            labels.setdefault(str(label), [])
            for key in keys:
                if str(key) not in labels[str(label)]:
                    labels[str(label)].append(str(key))
        self.expansions_by_key = expansions
        self.label_to_keys = {label: tuple(keys) for label, keys in labels.items()}

    def resolve_key(self, key_or_label: str) -> str:
        raw = str(key_or_label)
        if raw in self.expansions_by_key:
            return raw
        keys = tuple(self.label_to_keys.get(raw, ()))
        if len(keys) == 1:
            return str(keys[0])
        if len(keys) > 1:
            raise AlgebraicMetadataError(
                f"Algebraic label {raw!r} is ambiguous; use one of keys {keys!r}."
            )
        raise AlgebraicMetadataError(f"Unknown algebraic expansion key or label: {raw!r}.")

    def pair(self, lhs_key: str, rhs_key: str, *, coefficient_tol: float = 1.0e-12) -> AlgebraicPairMetadata:
        lhs = self.resolve_key(lhs_key)
        rhs = self.resolve_key(rhs_key)
        cache_key = (lhs, rhs, float(coefficient_tol))
        cached = self.pair_cache.get(cache_key)
        if cached is not None:
            return cached
        meta = build_pair_metadata(
            self.expansions_by_key[lhs],
            self.expansions_by_key[rhs],
            coefficient_tol=float(coefficient_tol),
        )
        self.pair_cache[cache_key] = meta
        return meta

    def summarize_local_context(
        self,
        candidate_key: str,
        context_keys: Sequence[str],
        *,
        coefficient_tol: float = 1.0e-12,
    ) -> AlgebraicLocalContextSummary:
        candidate = self.resolve_key(candidate_key)
        context = tuple(self.resolve_key(key) for key in context_keys)
        return summarize_local_context(
            self.expansions_by_key[candidate],
            [self.expansions_by_key[key] for key in context],
            pair_lookup=lambda lhs, rhs: self.pair(lhs, rhs, coefficient_tol=float(coefficient_tol)),
        )


def normalize_pauli_word_exyz(raw: Any, *, require_exyz: bool = True) -> str:
    """Normalize a Pauli word to lowercase repo-native e/x/y/z symbols."""

    word = str(raw).strip().lower()
    if require_exyz:
        invalid = sorted({ch for ch in word if ch not in {"e", "x", "y", "z"}})
        if invalid:
            raise AlgebraicMetadataError(
                f"Pauli word {raw!r} contains non e/x/y/z symbols: {invalid!r}."
            )
    return word


def support_qubits_from_pauli_word(word: str, *, nq: int | None = None) -> tuple[int, ...]:
    """Return support qubits using the repo convention qubit 0 = rightmost char."""

    normalized = normalize_pauli_word_exyz(word)
    n_qubits = len(normalized) if nq is None else int(nq)
    if len(normalized) != n_qubits:
        raise AlgebraicMetadataError(
            f"Pauli word length {len(normalized)} does not match nq={n_qubits}: {normalized!r}."
        )
    return tuple(
        sorted(int(n_qubits - 1 - idx) for idx, ch in enumerate(normalized) if ch != "e")
    )


def pauli_words_commute(lhs: str, rhs: str) -> bool:
    """Exact parity commutation check for two single Pauli words."""

    left = normalize_pauli_word_exyz(lhs)
    right = normalize_pauli_word_exyz(rhs)
    if len(left) != len(right):
        raise AlgebraicMetadataError(
            f"Pauli words must have equal length, got {len(left)} and {len(right)}."
        )
    anticommutes = 0
    for a, b in zip(left, right):
        if a == "e" or b == "e" or a == b:
            continue
        anticommutes += 1
    return bool(anticommutes % 2 == 0)


_PAULI_PRODUCT: dict[tuple[str, str], tuple[str, complex]] = {
    ("e", "e"): ("e", 1.0),
    ("e", "x"): ("x", 1.0),
    ("e", "y"): ("y", 1.0),
    ("e", "z"): ("z", 1.0),
    ("x", "e"): ("x", 1.0),
    ("y", "e"): ("y", 1.0),
    ("z", "e"): ("z", 1.0),
    ("x", "x"): ("e", 1.0),
    ("y", "y"): ("e", 1.0),
    ("z", "z"): ("e", 1.0),
    ("x", "y"): ("z", 1.0j),
    ("y", "x"): ("z", -1.0j),
    ("y", "z"): ("x", 1.0j),
    ("z", "y"): ("x", -1.0j),
    ("z", "x"): ("y", 1.0j),
    ("x", "z"): ("y", -1.0j),
}


def multiply_pauli_words(lhs: str, rhs: str) -> tuple[str, complex]:
    """Multiply two Pauli words and return ``(word, phase)``."""

    left = normalize_pauli_word_exyz(lhs)
    right = normalize_pauli_word_exyz(rhs)
    if len(left) != len(right):
        raise AlgebraicMetadataError(
            f"Pauli words must have equal length, got {len(left)} and {len(right)}."
        )
    out: list[str] = []
    phase = complex(1.0)
    for a, b in zip(left, right):
        ch, local_phase = _PAULI_PRODUCT[(a, b)]
        out.append(ch)
        phase *= complex(local_phase)
    return "".join(out), phase


def _terms_from_polynomial(polynomial: Any) -> tuple[SerializedPauliExpansionTerm, ...]:
    if polynomial is None or not hasattr(polynomial, "return_polynomial"):
        raise AlgebraicMetadataError("Ansatz term is missing a PauliPolynomial-like polynomial.")
    out: list[SerializedPauliExpansionTerm] = []
    for term in polynomial.return_polynomial():
        coeff = complex(term.p_coeff)
        out.append(
            SerializedPauliExpansionTerm(
                pauli_exyz=normalize_pauli_word_exyz(term.pw2strng()),
                coeff_re=float(coeff.real),
                coeff_im=float(coeff.imag),
                nq=int(term.nqubit()),
            )
        )
    if not out:
        raise AlgebraicMetadataError("Ansatz polynomial has no Pauli terms for exact metadata.")
    return tuple(out)


def _support_from_terms(terms: Sequence[SerializedPauliExpansionTerm]) -> tuple[int, ...]:
    support: set[int] = set()
    for term in terms:
        support.update(support_qubits_from_pauli_word(term.pauli_exyz, nq=int(term.nq)))
    return tuple(sorted(int(q) for q in support))


def _meta_get(meta: Any, key: str, default: Any = None) -> Any:
    if isinstance(meta, Mapping):
        return meta.get(key, default)
    return getattr(meta, key, default)


def _is_nonstring_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _serialized_terms_available(meta: Any) -> bool:
    compile_meta = _meta_get(meta, "compile_metadata", {})
    if not isinstance(compile_meta, Mapping):
        return False
    raw_terms = compile_meta.get("serialized_terms_exyz")
    if raw_terms is None:
        return False
    if not _is_nonstring_sequence(raw_terms):
        raise AlgebraicMetadataError("compile_metadata.serialized_terms_exyz must be a sequence of term mappings.")
    return bool(len(raw_terms) > 0)



def expansion_from_generator_metadata(
    meta: Any,
    *,
    key: str | None = None,
    label: str | None = None,
    require_exact: bool = True,
) -> GeneratorAlgebraicExpansion:
    """Build expansion metadata from ``GeneratorMetadata``/dict serialized terms.

    With ``require_exact=True`` this raises if ``compile_metadata.serialized_terms_exyz``
    is unavailable or empty.  That is the canonical repo-native lane policy.
    """

    compile_meta = _meta_get(meta, "compile_metadata", {})
    if not isinstance(compile_meta, Mapping):
        compile_meta = {}
    raw_terms = compile_meta.get("serialized_terms_exyz")
    candidate_label = str(label or _meta_get(meta, "candidate_label", key or ""))
    generator_id_raw = _meta_get(meta, "generator_id", None)
    generator_id = str(generator_id_raw) if generator_id_raw is not None else None
    expansion_key = str(key or generator_id or candidate_label)
    if not _is_nonstring_sequence(raw_terms) or len(raw_terms) == 0:
        if require_exact:
            raise AlgebraicMetadataError(
                f"Generator {candidate_label!r} is missing exact compile_metadata.serialized_terms_exyz."
            )
        support_raw = _meta_get(meta, "support_qubits", ())
        support = tuple(sorted(int(q) for q in support_raw)) if _is_nonstring_sequence(support_raw) else ()
        return GeneratorAlgebraicExpansion(
            key=expansion_key,
            label=candidate_label,
            generator_id=generator_id,
            terms=(),
            support_qubits=support,
            exactness=EXACTNESS_APPROX,
            source="missing_serialized_terms_approx",
        )
    terms: list[SerializedPauliExpansionTerm] = []
    for raw in raw_terms:
        if not isinstance(raw, Mapping):
            raise AlgebraicMetadataError(
                f"Generator {candidate_label!r} has a non-mapping serialized Pauli term: {raw!r}."
            )
        terms.append(SerializedPauliExpansionTerm.from_mapping(raw))
    return GeneratorAlgebraicExpansion(
        key=expansion_key,
        label=candidate_label,
        generator_id=generator_id,
        terms=tuple(terms),
        support_qubits=_support_from_terms(terms),
        exactness=EXACTNESS_EXACT,
        source="registry_serialized_terms",
    )


def expansion_from_ansatz_term(term: Any, *, key: str | None = None) -> GeneratorAlgebraicExpansion:
    """Extract exact metadata directly from an ``AnsatzTerm.polynomial``."""

    label = str(getattr(term, "label", key or ""))
    terms = _terms_from_polynomial(getattr(term, "polynomial", None))
    expansion_key = str(key or label)
    return GeneratorAlgebraicExpansion(
        key=expansion_key,
        label=label,
        generator_id=None,
        terms=terms,
        support_qubits=_support_from_terms(terms),
        exactness=EXACTNESS_EXACT,
        source="ansatz_term_polynomial",
    )


def _commutator_l1_norm_from_terms(
    lhs_terms: Sequence[SerializedPauliExpansionTerm],
    rhs_terms: Sequence[SerializedPauliExpansionTerm],
) -> float:
    accum: dict[str, complex] = {}
    for lhs in lhs_terms:
        for rhs in rhs_terms:
            if int(lhs.nq) != int(rhs.nq):
                raise AlgebraicMetadataError(
                    f"Cannot commute expansions with inconsistent nq values: {lhs.nq} vs {rhs.nq}."
                )
            lr_word, lr_phase = multiply_pauli_words(lhs.pauli_exyz, rhs.pauli_exyz)
            rl_word, rl_phase = multiply_pauli_words(rhs.pauli_exyz, lhs.pauli_exyz)
            coeff = lhs.coeff * rhs.coeff
            accum[lr_word] = accum.get(lr_word, 0.0j) + coeff * lr_phase
            accum[rl_word] = accum.get(rl_word, 0.0j) - coeff * rl_phase
    return float(sum(abs(val) for val in accum.values()))


def exact_expansions_commute(
    lhs: GeneratorAlgebraicExpansion,
    rhs: GeneratorAlgebraicExpansion,
    *,
    coefficient_tol: float = 1.0e-12,
) -> tuple[bool, float]:
    """Check full-polynomial commutator exactness, not pairwise term commutation."""

    if lhs.exactness != EXACTNESS_EXACT or rhs.exactness != EXACTNESS_EXACT:
        raise AlgebraicMetadataError("Exact commutation requires exact expansions for both generators.")
    if not lhs.terms or not rhs.terms:
        raise AlgebraicMetadataError("Exact commutation requires non-empty Pauli expansions.")
    l1_norm = _commutator_l1_norm_from_terms(lhs.terms, rhs.terms)
    return bool(l1_norm <= float(coefficient_tol)), float(l1_norm)


def build_pair_metadata(
    lhs: GeneratorAlgebraicExpansion,
    rhs: GeneratorAlgebraicExpansion,
    *,
    coefficient_tol: float = 1.0e-12,
) -> AlgebraicPairMetadata:
    """Build exact pair metadata for lane assignment/prune windows."""

    overlap_qubits = tuple(sorted(set(lhs.support_qubits).intersection(rhs.support_qubits)))
    support_overlap = bool(overlap_qubits)
    if lhs.exactness != EXACTNESS_EXACT or rhs.exactness != EXACTNESS_EXACT:
        return AlgebraicPairMetadata(
            lhs_key=str(lhs.key),
            rhs_key=str(rhs.key),
            lhs_label=str(lhs.label),
            rhs_label=str(rhs.label),
            support_overlap=bool(support_overlap),
            commutes=None,
            exactness=EXACTNESS_APPROX,
            relation=RELATION_APPROX_OR_UNKNOWN,
            support_overlap_qubits=overlap_qubits,
            commutator_l1_norm=None,
        )
    commutes, l1_norm = exact_expansions_commute(lhs, rhs, coefficient_tol=float(coefficient_tol))
    if support_overlap and commutes:
        relation = RELATION_FLAT_COMM
    elif support_overlap and not commutes:
        relation = RELATION_CURV_NONCOMM
    elif not support_overlap and commutes:
        relation = RELATION_DISJ_COMM
    else:
        relation = RELATION_CURV_NONCOMM
    return AlgebraicPairMetadata(
        lhs_key=str(lhs.key),
        rhs_key=str(rhs.key),
        lhs_label=str(lhs.label),
        rhs_label=str(rhs.label),
        support_overlap=bool(support_overlap),
        commutes=bool(commutes),
        exactness=EXACTNESS_EXACT,
        relation=str(relation),
        support_overlap_qubits=overlap_qubits,
        commutator_l1_norm=float(l1_norm),
    )


def assign_lane(
    *,
    n_flat: int = 0,
    n_curv: int = 0,
    n_disj: int = 0,
    n_approx: int = 0,
) -> str:
    """Assign Phase-1 algebraic lane from exact local-context counts."""

    flat = int(n_flat)
    curv = int(n_curv)
    disj = int(n_disj)
    approx = int(n_approx)
    if approx > 0:
        return LANE_MIX
    if flat > 0 and curv == 0:
        return LANE_FLAT
    if curv > 0 and flat == 0:
        return LANE_CURV
    if flat == 0 and curv == 0 and disj > 0:
        return LANE_DISJ
    return LANE_MIX


def summarize_local_context(
    candidate: GeneratorAlgebraicExpansion,
    context: Sequence[GeneratorAlgebraicExpansion],
    *,
    pair_lookup: Callable[[str, str], AlgebraicPairMetadata] | None = None,
) -> AlgebraicLocalContextSummary:
    """Summarize exact pair relations between one candidate and local context."""

    n_flat = 0
    n_curv = 0
    n_disj = 0
    n_approx = 0
    labels: list[str] = []
    keys: list[str] = []
    for other in context:
        labels.append(str(other.label))
        keys.append(str(other.key))
        pair = pair_lookup(str(candidate.key), str(other.key)) if pair_lookup is not None else build_pair_metadata(candidate, other)
        if pair.exactness != EXACTNESS_EXACT or pair.relation == RELATION_APPROX_OR_UNKNOWN:
            n_approx += 1
        elif pair.relation == RELATION_FLAT_COMM:
            n_flat += 1
        elif pair.relation == RELATION_CURV_NONCOMM:
            n_curv += 1
        elif pair.relation == RELATION_DISJ_COMM:
            n_disj += 1
        else:
            n_approx += 1
    lane = assign_lane(n_flat=n_flat, n_curv=n_curv, n_disj=n_disj, n_approx=n_approx)
    return AlgebraicLocalContextSummary(
        candidate_key=str(candidate.key),
        candidate_label=str(candidate.label),
        context_keys=tuple(keys),
        context_labels=tuple(labels),
        n_flat=int(n_flat),
        n_curv=int(n_curv),
        n_disj=int(n_disj),
        n_approx=int(n_approx),
        lane=str(lane),
        quality=EXACTNESS_EXACT if n_approx == 0 else EXACTNESS_APPROX,
    )


def _finite_record_score(record: Mapping[str, Any], key: str | None, default: float = float("-inf")) -> float:
    if key is None:
        return 0.0
    raw = record.get(str(key), default)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _normalized_lanes(lanes: Sequence[str] | None) -> tuple[str, ...]:
    out = tuple(str(lane) for lane in (LANES_PHASE1 if lanes is None else lanes))
    return out if out else tuple(str(lane) for lane in LANES_PHASE1)


def _record_lane(
    record: Mapping[str, Any],
    lane_key: str,
    *,
    lanes: Sequence[str] | None = None,
    fallback_lane: str = LANE_MIX,
) -> str:
    lane_choices = _normalized_lanes(lanes)
    fallback = str(fallback_lane)
    lane = str(record.get(str(lane_key), fallback))
    return lane if lane in lane_choices else fallback


def _record_rank_key(
    record: Mapping[str, Any],
    *,
    score_key: str,
    tie_break_score_key: str | None = None,
) -> tuple[float, float, int, int]:
    return (
        -_finite_record_score(record, score_key),
        -_finite_record_score(record, tie_break_score_key),
        int(record.get("candidate_pool_index", -1)),
        int(record.get("position_id", -1)),
    )


def _lane_budget_allocation(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    cap: int,
    lane_budgets: Mapping[str, int] | None = None,
    lanes: Sequence[str] | None = None,
) -> dict[str, int]:
    cap_eff = int(max(0, cap))
    lane_choices = _normalized_lanes(lanes)
    budgets = {lane: 0 for lane in lane_choices}
    if cap_eff <= 0:
        return budgets
    if lane_budgets is not None:
        for lane in lane_choices:
            available = len(grouped.get(lane, ()))
            if available > 0:
                budgets[lane] = int(min(available, max(0, lane_budgets.get(lane, 0))))
        total = sum(budgets.values())
        if total > cap_eff:
            remaining = cap_eff
            for lane in lane_choices:
                keep = min(int(budgets[lane]), int(remaining))
                budgets[lane] = int(keep)
                remaining -= int(keep)
        return budgets

    nonempty = [lane for lane in lane_choices if len(grouped.get(lane, ())) > 0]
    remaining = cap_eff
    for lane in nonempty:
        if remaining <= 0:
            break
        budgets[lane] = 1
        remaining -= 1
    while remaining > 0 and nonempty:
        progressed = False
        for lane in sorted(nonempty, key=lambda key: (-len(grouped.get(key, ())), lane_choices.index(key))):
            if budgets[lane] >= len(grouped.get(lane, ())) or remaining <= 0:
                continue
            budgets[lane] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return budgets


def algebraic_lane_quota_pressure_budgets(
    records: Sequence[Mapping[str, Any]],
    *,
    cap: int,
    score_key: str,
    threshold: float = float("-inf"),
    pressure: float = 1.0,
    lane_key: str = "algebraic_lane",
    lane_abs_threshold: float | None = None,
    lane_rel_threshold: float = 0.0,
    tie_break_score_key: str | None = None,
    lanes: Sequence[str] | None = None,
    fallback_lane: str = LANE_MIX,
) -> dict[str, int]:
    """Allocate deterministic lane budgets from one quota-pressure scalar.

    ``pressure=1`` reserves one slot for every live lane when the cap allows.
    Lower pressure reserves fewer lanes, then globally refills by score so the
    strongest lanes can consume more of the shortlist cap.  Returned budgets
    always sum to ``min(cap, eligible_count)`` unless there are no eligible
    records, and no lane budget exceeds that lane's available records.
    """

    cap_eff = int(max(0, cap))
    lane_choices = _normalized_lanes(lanes)
    fallback = str(fallback_lane)
    budgets = {lane: 0 for lane in lane_choices}
    if cap_eff <= 0:
        return budgets
    eligible = [
        dict(record)
        for record in records
        if _finite_record_score(record, score_key) >= float(threshold)
    ]
    if not eligible:
        eligible = [dict(record) for record in records]
    if not eligible:
        return budgets

    grouped: dict[str, list[dict[str, Any]]] = {lane: [] for lane in lane_choices}
    for record in eligible:
        grouped[_record_lane(record, lane_key, lanes=lane_choices, fallback_lane=fallback)].append(dict(record))
    for lane in lane_choices:
        grouped[lane] = sorted(
            grouped[lane],
            key=lambda record: _record_rank_key(
                record,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            ),
        )

    lane_health = {
        lane: (_finite_record_score(grouped[lane][0], score_key, default=0.0) if grouped[lane] else 0.0)
        for lane in lane_choices
    }
    health_star = max(lane_health.values()) if lane_health else 0.0
    abs_floor = float(threshold if lane_abs_threshold is None else lane_abs_threshold)
    rel_floor = float(max(0.0, min(1.0, lane_rel_threshold)))
    live_lanes: list[str] = []
    for lane in lane_choices:
        if not grouped[lane]:
            continue
        rel = float(lane_health[lane] / health_star) if health_star > 0.0 else 0.0
        if lane_health[lane] >= abs_floor and rel >= rel_floor:
            live_lanes.append(lane)
    if not live_lanes:
        return budgets

    live_records = [row for lane in live_lanes for row in grouped[lane]]
    target = int(min(cap_eff, len(live_records)))
    if target <= 0:
        return budgets

    ranked_lanes = sorted(
        live_lanes,
        key=lambda lane: _record_rank_key(
            grouped[lane][0],
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        ),
    )
    pressure_val = float(max(0.0, min(1.0, pressure)))
    reserved_lane_count = int(math.ceil(pressure_val * len(ranked_lanes)))
    reserved_lane_count = int(min(target, len(ranked_lanes), max(0, reserved_lane_count)))
    for lane in ranked_lanes[:reserved_lane_count]:
        budgets[lane] = 1

    remaining = int(target - sum(budgets.values()))
    globally_ranked = sorted(
        (
            record
            for lane in live_lanes
            for record in grouped[lane][int(budgets.get(lane, 0)) :]
        ),
        key=lambda record: _record_rank_key(
            record,
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        ),
    )
    for record in globally_ranked:
        if remaining <= 0:
            break
        lane = _record_lane(record, lane_key, lanes=lane_choices, fallback_lane=fallback)
        if lane not in live_lanes:
            continue
        if budgets[lane] >= len(grouped[lane]):
            continue
        budgets[lane] += 1
        remaining -= 1
    return budgets


def lane_quota_pressure_budgets(
    records: Sequence[Mapping[str, Any]],
    *,
    cap: int,
    score_key: str,
    threshold: float = float("-inf"),
    pressure: float = 1.0,
    lane_key: str,
    lanes: Sequence[str],
    fallback_lane: str,
    lane_abs_threshold: float | None = None,
    lane_rel_threshold: float = 0.0,
    tie_break_score_key: str | None = None,
) -> dict[str, int]:
    return algebraic_lane_quota_pressure_budgets(
        records,
        cap=cap,
        score_key=score_key,
        threshold=threshold,
        pressure=pressure,
        lane_key=lane_key,
        lanes=lanes,
        fallback_lane=fallback_lane,
        lane_abs_threshold=lane_abs_threshold,
        lane_rel_threshold=lane_rel_threshold,
        tie_break_score_key=tie_break_score_key,
    )


def _record_identity(record: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(record.get("candidate_label", "")),
        str(record.get("generator_id", "")),
        int(record.get("candidate_pool_index", -1)),
        int(record.get("position_id", -1)),
    )


def _global_refill_shortlist(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    cap: int,
    tie_break_score_key: str | None = None,
) -> list[dict[str, Any]]:
    cap_eff = int(max(0, cap))
    out = [dict(row) for row in selected]
    if cap_eff <= 0:
        return []
    target = int(min(cap_eff, len(candidates)))
    if len(out) >= target:
        return out[:target]
    seen = {_record_identity(row) for row in out}
    for row in sorted(
        [dict(candidate) for candidate in candidates],
        key=lambda record: _record_rank_key(
            record,
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        ),
    ):
        key = _record_identity(row)
        if key in seen:
            continue
        out.append(dict(row))
        seen.add(key)
        if len(out) >= target:
            break
    return out


def _frontier_take(
    ranked: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    cap: int,
    frontier_ratio: float,
    score_eps: float = 1.0e-12,
) -> list[dict[str, Any]]:
    if int(cap) <= 0 or not ranked:
        return []
    cap_eff = int(max(1, min(int(cap), len(ranked))))
    take = int(cap_eff)
    frontier_cut = float(max(0.0, min(1.0, frontier_ratio)))
    if cap_eff > 1 and 0.0 < frontier_cut < 1.0:
        for idx in range(cap_eff - 1):
            s_cur = _finite_record_score(ranked[idx], score_key, default=0.0)
            s_next = _finite_record_score(ranked[idx + 1], score_key, default=0.0)
            ratio = float((s_next + float(score_eps)) / (s_cur + float(score_eps)))
            if ratio <= frontier_cut:
                take = int(idx + 1)
                break
    return [dict(row) for row in ranked[:take]]


def _mark_shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    shortlist_flag: str | None,
    feature_updater: Callable[[Any, Mapping[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    size = int(len(records))
    for rank, record in enumerate(records, start=1):
        updated = dict(record)
        updated["shortlist_rank"] = int(rank)
        updated["shortlist_size"] = int(size)
        if shortlist_flag is not None:
            updated[str(shortlist_flag)] = True
        feat = updated.get("feature")
        if feature_updater is not None and feat is not None:
            payload: dict[str, Any] = {
                "shortlist_rank": int(rank),
                "shortlist_size": int(size),
            }
            if shortlist_flag is not None:
                payload[str(shortlist_flag)] = True
            updated["feature"] = feature_updater(feat, payload)
        out.append(updated)
    return out


def phase1_lane_shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    score_key: str = "simple_score",
    threshold: float = float("-inf"),
    cap: int,
    frontier_ratio: float,
    lane_key: str = "algebraic_lane",
    lane_budgets: Mapping[str, int] | None = None,
    tie_break_score_key: str | None = None,
    shortlist_flag: str | None = None,
    feature_updater: Callable[[Any, Mapping[str, Any]], Any] | None = None,
    lanes: Sequence[str] | None = None,
    fallback_lane: str = LANE_MIX,
) -> list[dict[str, Any]]:
    """Lane-wise Phase-1 shortlist with geometry-only ranking inside lanes."""

    cap_eff = int(max(0, cap))
    if cap_eff <= 0:
        return []
    filtered = [
        dict(record)
        for record in records
        if _finite_record_score(record, score_key) >= float(threshold)
    ]
    if not filtered:
        filtered = [dict(record) for record in records]
    lane_choices = _normalized_lanes(lanes)
    fallback = str(fallback_lane)
    grouped: dict[str, list[dict[str, Any]]] = {lane: [] for lane in lane_choices}
    for record in filtered:
        grouped[_record_lane(record, lane_key, lanes=lane_choices, fallback_lane=fallback)].append(dict(record))
    for lane in lane_choices:
        grouped[lane] = sorted(
            grouped[lane],
            key=lambda record: _record_rank_key(
                record,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            ),
        )
    budgets = _lane_budget_allocation(
        grouped,
        cap=cap_eff,
        lane_budgets=lane_budgets,
        lanes=lane_choices,
    )
    selected: list[dict[str, Any]] = []
    for lane in lane_choices:
        selected.extend(
            _frontier_take(
                grouped[lane],
                score_key=score_key,
                cap=int(budgets.get(lane, 0)),
                frontier_ratio=float(frontier_ratio),
            )
        )
    selected = _global_refill_shortlist(
        selected,
        filtered,
        score_key=score_key,
        cap=cap_eff,
        tie_break_score_key=tie_break_score_key,
    )
    if not selected and filtered:
        selected = [
            sorted(
                filtered,
                key=lambda record: _record_rank_key(
                    record,
                    score_key=score_key,
                    tie_break_score_key=tie_break_score_key,
                ),
            )[0]
        ]
    selected = sorted(
        selected,
        key=lambda record: _record_rank_key(
            record,
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        ),
    )[:cap_eff]
    return _mark_shortlist_records(
        selected,
        shortlist_flag=shortlist_flag,
        feature_updater=feature_updater,
    )


def lane_phase1_shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    score_key: str = "simple_score",
    threshold: float = float("-inf"),
    cap: int,
    frontier_ratio: float,
    lane_key: str,
    lanes: Sequence[str],
    fallback_lane: str,
    lane_budgets: Mapping[str, int] | None = None,
    tie_break_score_key: str | None = None,
    shortlist_flag: str | None = None,
    feature_updater: Callable[[Any, Mapping[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    return phase1_lane_shortlist_records(
        records,
        score_key=score_key,
        threshold=threshold,
        cap=cap,
        frontier_ratio=frontier_ratio,
        lane_key=lane_key,
        lanes=lanes,
        fallback_lane=fallback_lane,
        lane_budgets=lane_budgets,
        tie_break_score_key=tie_break_score_key,
        shortlist_flag=shortlist_flag,
        feature_updater=feature_updater,
    )


def _phase0_weak_counts_payload(
    *,
    lane: str,
    quality: str,
    n_flat: int = 0,
    n_curv: int = 0,
    n_disj: int = 0,
    n_approx: int = 0,
    context_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "phase0_algebraic_lane": str(lane if lane in LANES_PHASE1 else LANE_MIX),
        "phase0_algebraic_quality": str(quality),
        "phase0_algebraic_context_counts": {
            "n_flat": int(n_flat),
            "n_curv": int(n_curv),
            "n_disj": int(n_disj),
            "n_approx": int(n_approx),
        },
        "phase0_algebraic_context_labels": [str(x) for x in (context_labels or [])],
    }


PHASE0_WEAK_ALGEBRAIC_SCHEMA = "phase0_weak_algebraic_lanes_v1"


def _phase0_approx_expansion(
    *,
    key: str,
    label: str,
    generator_id: str | None,
    support_qubits: Sequence[int] | None,
    source: str,
) -> GeneratorAlgebraicExpansion:
    support: tuple[int, ...]
    try:
        support = tuple(sorted({int(q) for q in (support_qubits or [])}))
    except (TypeError, ValueError):
        support = ()
    return GeneratorAlgebraicExpansion(
        key=str(key),
        label=str(label),
        generator_id=(None if generator_id is None else str(generator_id)),
        terms=(),
        support_qubits=tuple(support),
        exactness=EXACTNESS_APPROX,
        source=str(source),
    )


def _phase0_meta_support(meta: Any) -> tuple[int, ...]:
    raw_support = _meta_get(meta, "support_qubits", ())
    if _is_nonstring_sequence(raw_support):
        try:
            return tuple(sorted({int(q) for q in raw_support}))
        except (TypeError, ValueError):
            return ()
    return ()


def _phase0_weak_expansion_for_term(
    term: Any | None,
    meta: Any | None,
    *,
    label: str,
    allow_polynomial_source: bool,
) -> tuple[GeneratorAlgebraicExpansion, str | None]:
    candidate_label = str(label or _meta_get(meta, "candidate_label", getattr(term, "label", "")))
    generator_id_raw = _meta_get(meta, "generator_id", None)
    generator_id = str(generator_id_raw) if generator_id_raw is not None else None
    expansion_key = str(generator_id or candidate_label)
    malformed_reason: str | None = None
    serialized_available = False
    if meta is not None:
        try:
            serialized_available = bool(_serialized_terms_available(meta))
        except AlgebraicMetadataError as exc:
            malformed_reason = str(exc)
            serialized_available = False
        if malformed_reason is not None and not serialized_available:
            return _phase0_approx_expansion(
                key=expansion_key,
                label=candidate_label,
                generator_id=generator_id,
                support_qubits=_phase0_meta_support(meta),
                source="phase0_malformed_serialized_terms_approx",
            ), malformed_reason
        if serialized_available:
            try:
                return expansion_from_generator_metadata(
                    meta,
                    key=expansion_key,
                    label=candidate_label,
                    require_exact=True,
                ), None
            except AlgebraicMetadataError as exc:
                malformed_reason = str(exc)
                return _phase0_approx_expansion(
                    key=expansion_key,
                    label=candidate_label,
                    generator_id=generator_id,
                    support_qubits=_phase0_meta_support(meta),
                    source="phase0_malformed_serialized_terms_approx",
                ), malformed_reason
    if bool(allow_polynomial_source) and term is not None and hasattr(term, "polynomial"):
        try:
            return expansion_from_ansatz_term(term, key=candidate_label), None
        except AlgebraicMetadataError as exc:
            malformed_reason = str(exc)
    if meta is not None:
        try:
            approx = expansion_from_generator_metadata(
                meta,
                key=expansion_key,
                label=candidate_label,
                require_exact=False,
            )
            return approx, malformed_reason
        except AlgebraicMetadataError as exc:
            malformed_reason = str(exc)
    return _phase0_approx_expansion(
        key=expansion_key,
        label=candidate_label,
        generator_id=generator_id,
        support_qubits=_phase0_meta_support(meta),
        source="phase0_missing_metadata_approx",
    ), malformed_reason


def build_phase0_weak_algebraic_index(
    *,
    pool: Sequence[Any] | None = None,
    registry_by_label: Mapping[str, Any] | None = None,
    allow_polynomial_source: bool = True,
) -> tuple[AlgebraicMetadataIndex, dict[str, Any]]:
    """Build a permissive weak Phase-0 algebraic index.

    Unlike ``build_exact_expansion_index()``, this helper never raises for
    missing/malformed generator metadata.  Bad or absent metadata is represented
    as approximate metadata, which forces downstream lane assignment to ``mix``.
    """

    registry = dict(registry_by_label or {})
    expansions: dict[str, GeneratorAlgebraicExpansion] = {}
    label_to_keys: dict[str, list[str]] = {}
    errors: list[str] = []
    degraded_keys: set[str] = set()

    def _add(expansion: GeneratorAlgebraicExpansion) -> None:
        key = str(expansion.key)
        if key in expansions and expansions[key] != expansion:
            suffix = 2
            base = key
            while f"{base}#{suffix}" in expansions:
                suffix += 1
            key = f"{base}#{suffix}"
            expansion = GeneratorAlgebraicExpansion(
                key=key,
                label=expansion.label,
                generator_id=expansion.generator_id,
                terms=expansion.terms,
                support_qubits=expansion.support_qubits,
                exactness=expansion.exactness,
                source=expansion.source,
            )
        expansions[key] = expansion
        label_to_keys.setdefault(str(expansion.label), []).append(key)

    if pool is not None:
        for term in pool:
            label = str(getattr(term, "label", ""))
            meta = registry.get(label)
            expansion, error = _phase0_weak_expansion_for_term(
                term,
                meta,
                label=label,
                allow_polynomial_source=bool(allow_polynomial_source),
            )
            if error is not None:
                errors.append(f"{label}: {error}")
            if error is not None or expansion.exactness != EXACTNESS_EXACT:
                degraded_keys.add(str(expansion.key))
            _add(expansion)
    else:
        for label, meta in registry.items():
            expansion, error = _phase0_weak_expansion_for_term(
                None,
                meta,
                label=str(label),
                allow_polynomial_source=False,
            )
            if error is not None:
                errors.append(f"{label}: {error}")
            if error is not None or expansion.exactness != EXACTNESS_EXACT:
                degraded_keys.add(str(expansion.key))
            _add(expansion)

    index = AlgebraicMetadataIndex(
        expansions_by_key=expansions,
        label_to_keys={label: tuple(keys) for label, keys in label_to_keys.items()},
    )
    exact_count = sum(1 for exp in index.expansions_by_key.values() if exp.exactness == EXACTNESS_EXACT)
    approx_count = sum(1 for exp in index.expansions_by_key.values() if exp.exactness != EXACTNESS_EXACT)
    summary = {
        "schema": PHASE0_WEAK_ALGEBRAIC_SCHEMA,
        "status": "ready",
        "strict_exact": False,
        "expansion_count": int(len(index.expansions_by_key)),
        "exact_count": int(exact_count),
        "approx_count": int(approx_count),
        "error_count": int(len(errors)),
        "degraded_count": int(len(degraded_keys)),
        "errors": [str(x) for x in errors[:20]],
    }
    return index, summary


def phase0_weak_lane_payload(
    index: AlgebraicMetadataIndex | None,
    *,
    candidate_label: str,
    context_labels: Sequence[str] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Return weak Phase-0 lane telemetry; missing metadata degrades to ``mix``."""

    labels = [str(x) for x in (context_labels or []) if str(x)]
    if not bool(enabled):
        return _phase0_weak_counts_payload(
            lane=LANE_MIX,
            quality="inactive",
            context_labels=labels,
        )
    if index is None:
        return _phase0_weak_counts_payload(
            lane=LANE_MIX,
            quality="index_unavailable",
            n_approx=max(1, len(labels)) if labels else 0,
            context_labels=labels,
        )
    try:
        index.resolve_key(str(candidate_label))
    except AlgebraicMetadataError:
        return _phase0_weak_counts_payload(
            lane=LANE_MIX,
            quality="missing_candidate_metadata",
            n_approx=max(1, len(labels)) if labels else 1,
            context_labels=labels,
        )
    resolvable: list[str] = []
    missing_count = 0
    for label in labels:
        try:
            index.resolve_key(str(label))
        except AlgebraicMetadataError:
            missing_count += 1
            continue
        resolvable.append(str(label))
    if not resolvable:
        return _phase0_weak_counts_payload(
            lane=LANE_MIX,
            quality=("empty_context" if missing_count == 0 else "missing_context_metadata"),
            n_approx=int(missing_count),
            context_labels=labels,
        )
    try:
        summary = index.summarize_local_context(str(candidate_label), resolvable)
    except AlgebraicMetadataError:
        return _phase0_weak_counts_payload(
            lane=LANE_MIX,
            quality="malformed_context_metadata",
            n_approx=max(1, len(labels)),
            context_labels=labels,
        )
    n_approx_total = int(summary.n_approx) + int(missing_count)
    lane = assign_lane(
        n_flat=int(summary.n_flat),
        n_curv=int(summary.n_curv),
        n_disj=int(summary.n_disj),
        n_approx=int(n_approx_total),
    )
    quality = EXACTNESS_EXACT if n_approx_total == 0 else EXACTNESS_APPROX
    return _phase0_weak_counts_payload(
        lane=str(lane),
        quality=str(quality),
        n_flat=int(summary.n_flat),
        n_curv=int(summary.n_curv),
        n_disj=int(summary.n_disj),
        n_approx=int(n_approx_total),
        context_labels=labels,
    )


def phase0_lane_shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    score_key: str = "phase0_delta_e_upper_hw",
    threshold: float = 0.0,
    cap: int = 0,
    pressure: float = 1.0,
    lane_key: str = "phase0_algebraic_lane",
    tie_break_score_key: str | None = "phase0_raw_gradient_abs",
    shortlist_flag: str | None = "phase0_pilot_retained",
) -> list[dict[str, Any]]:
    """Weak Phase-0 lane shortlist; ``cap=0`` means uncapped/no forced pruning."""

    records_list = [dict(rec) for rec in records]
    if not records_list:
        return []
    threshold_val = float(threshold)
    eligible = [
        dict(record)
        for record in records_list
        if _finite_record_score(record, score_key) >= threshold_val
    ]
    fallback_reason: str | None = None
    if not eligible:
        fallback_reason = "fallback_strongest_after_threshold"
        eligible = [
            sorted(
                records_list,
                key=lambda record: _record_rank_key(
                    record,
                    score_key=score_key,
                    tie_break_score_key=tie_break_score_key,
                ),
            )[0]
        ]
    cap_eff = int(max(0, cap))
    if cap_eff <= 0:
        selected = sorted(
            [dict(row) for row in eligible],
            key=lambda record: _record_rank_key(
                record,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            ),
        )
        budgets = {
            lane: sum(1 for row in selected if _record_lane(row, lane_key) == lane)
            for lane in LANES_PHASE1
        }
    else:
        budgets = algebraic_lane_quota_pressure_budgets(
            eligible,
            cap=cap_eff,
            score_key=score_key,
            threshold=threshold_val,
            pressure=float(pressure),
            lane_key=lane_key,
            tie_break_score_key=tie_break_score_key,
        )
        grouped: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANES_PHASE1}
        for record in eligible:
            grouped[_record_lane(record, lane_key)].append(dict(record))
        for lane in LANES_PHASE1:
            grouped[lane] = sorted(
                grouped[lane],
                key=lambda record: _record_rank_key(
                    record,
                    score_key=score_key,
                    tie_break_score_key=tie_break_score_key,
                ),
            )
        selected = []
        for lane in LANES_PHASE1:
            selected.extend(grouped[lane][: int(max(0, budgets.get(lane, 0)))])
        selected = _global_refill_shortlist(
            selected,
            eligible,
            score_key=score_key,
            cap=cap_eff,
            tie_break_score_key=tie_break_score_key,
        )
        selected = sorted(
            selected,
            key=lambda record: _record_rank_key(
                record,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            ),
        )[:cap_eff]
    lane_sizes = {
        lane: sum(1 for row in eligible if _record_lane(row, lane_key) == lane)
        for lane in LANES_PHASE1
    }
    lane_ranks: dict[str, int] = {lane: 0 for lane in LANES_PHASE1}
    marked: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        lane = _record_lane(row, lane_key)
        lane_ranks[lane] = int(lane_ranks.get(lane, 0) + 1)
        updated = dict(row)
        updated["phase0_pilot_rank"] = int(rank)
        updated["phase0_pilot_size"] = int(len(selected))
        updated["phase0_lane_rank"] = int(lane_ranks[lane])
        updated["phase0_lane_size"] = int(lane_sizes.get(lane, 0))
        updated["phase0_lane_budget"] = int(budgets.get(lane, 0))
        updated["phase0_filter_reason"] = str(fallback_reason or "retained")
        if shortlist_flag is not None:
            updated[str(shortlist_flag)] = True
        marked.append(updated)
    return marked



def phase2_lane_health_shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    score_key: str = "phase2_raw_score",
    threshold: float = float("-inf"),
    cap: int,
    frontier_ratio: float,
    lane_key: str = "algebraic_lane",
    lane_abs_threshold: float | None = None,
    lane_rel_threshold: float = 0.0,
    lane_budgets: Mapping[str, int] | None = None,
    tie_break_score_key: str | None = None,
    shortlist_flag: str | None = None,
    feature_updater: Callable[[Any, Mapping[str, Any]], Any] | None = None,
    lanes: Sequence[str] | None = None,
    fallback_lane: str = LANE_MIX,
    health_key_prefix: str = "algebraic",
) -> list[dict[str, Any]]:
    """Phase-2 inherited-lane survival using geometry-only lane health."""

    cap_eff = int(max(0, cap))
    if cap_eff <= 0:
        return []
    lane_choices = _normalized_lanes(lanes)
    fallback = str(fallback_lane)
    health_prefix = str(health_key_prefix)
    lane_health_key = f"{health_prefix}_lane_health"
    lane_relative_health_key = f"{health_prefix}_lane_relative_health"
    lane_live_key = f"{health_prefix}_lane_live"
    grouped: dict[str, list[dict[str, Any]]] = {lane: [] for lane in lane_choices}
    for record in records:
        grouped[_record_lane(record, lane_key, lanes=lane_choices, fallback_lane=fallback)].append(dict(record))
    for lane in lane_choices:
        grouped[lane] = sorted(
            grouped[lane],
            key=lambda record: _record_rank_key(
                record,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            ),
        )
    lane_health = {
        lane: (_finite_record_score(grouped[lane][0], score_key, default=0.0) if grouped[lane] else 0.0)
        for lane in lane_choices
    }
    health_star = max(lane_health.values()) if lane_health else 0.0
    abs_floor = float(threshold if lane_abs_threshold is None else lane_abs_threshold)
    rel_floor = float(max(0.0, min(1.0, lane_rel_threshold)))
    live_lanes: list[str] = []
    for lane in lane_choices:
        rel = float(lane_health[lane] / health_star) if health_star > 0.0 else 0.0
        live = bool(grouped[lane] and lane_health[lane] >= abs_floor and rel >= rel_floor)
        if live:
            live_lanes.append(lane)
        for idx, row in enumerate(grouped[lane]):
            grouped[lane][idx] = {
                **dict(row),
                lane_health_key: float(lane_health[lane]),
                lane_relative_health_key: float(rel),
                lane_live_key: bool(live),
            }
    budgets = _lane_budget_allocation(
        {lane: grouped[lane] for lane in live_lanes},
        cap=cap_eff,
        lane_budgets=lane_budgets,
        lanes=lane_choices,
    )
    selected: list[dict[str, Any]] = []
    live_candidates: list[dict[str, Any]] = []
    for lane in lane_choices:
        if lane not in live_lanes:
            continue
        candidates = [
            row for row in grouped[lane]
            if _finite_record_score(row, score_key) >= float(threshold)
        ]
        live_candidates.extend(dict(row) for row in candidates)
        selected.extend(
            _frontier_take(
                candidates,
                score_key=score_key,
                cap=int(budgets.get(lane, 0)),
                frontier_ratio=float(frontier_ratio),
            )
        )
    selected = _global_refill_shortlist(
        selected,
        live_candidates,
        score_key=score_key,
        cap=cap_eff,
        tie_break_score_key=tie_break_score_key,
    )
    if not selected:
        fallback = sorted(
            [dict(row) for row in records],
            key=lambda record: _record_rank_key(
                record,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            ),
        )
        selected = fallback[:1]
    selected = sorted(
        selected,
        key=lambda record: _record_rank_key(
            record,
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        ),
    )[:cap_eff]
    marked: list[dict[str, Any]] = []
    for row in selected:
        updates = {
            lane_health_key: row.get(lane_health_key),
            lane_relative_health_key: row.get(lane_relative_health_key),
            lane_live_key: row.get(lane_live_key),
        }
        feat = row.get("feature")
        updated = dict(row)
        if feature_updater is not None and feat is not None:
            updated["feature"] = feature_updater(feat, updates)
        marked.append(updated)
    return _mark_shortlist_records(
        marked,
        shortlist_flag=shortlist_flag,
        feature_updater=feature_updater,
    )


def lane_phase2_health_shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    score_key: str = "phase2_raw_score",
    threshold: float = float("-inf"),
    cap: int,
    frontier_ratio: float,
    lane_key: str,
    lanes: Sequence[str],
    fallback_lane: str,
    lane_abs_threshold: float | None = None,
    lane_rel_threshold: float = 0.0,
    lane_budgets: Mapping[str, int] | None = None,
    tie_break_score_key: str | None = None,
    shortlist_flag: str | None = None,
    feature_updater: Callable[[Any, Mapping[str, Any]], Any] | None = None,
    health_key_prefix: str = "algebraic",
) -> list[dict[str, Any]]:
    return phase2_lane_health_shortlist_records(
        records,
        score_key=score_key,
        threshold=threshold,
        cap=cap,
        frontier_ratio=frontier_ratio,
        lane_key=lane_key,
        lanes=lanes,
        fallback_lane=fallback_lane,
        lane_abs_threshold=lane_abs_threshold,
        lane_rel_threshold=lane_rel_threshold,
        lane_budgets=lane_budgets,
        tie_break_score_key=tie_break_score_key,
        shortlist_flag=shortlist_flag,
        feature_updater=feature_updater,
        health_key_prefix=health_key_prefix,
    )


def build_exact_expansion_index(
    *,
    pool: Sequence[Any] | None = None,
    registry_by_label: Mapping[str, Any] | None = None,
    require_exact: bool = True,
    allow_polynomial_source: bool = True,
) -> AlgebraicMetadataIndex:
    """Build exact expansion index from registry metadata and/or Ansatz terms.

    Valid exact sources are ``compile_metadata.serialized_terms_exyz`` and, when
    explicitly allowed, direct ``AnsatzTerm.polynomial`` extraction.  Label-only
    metadata is never promoted to exact algebraic lane input.
    """

    registry = dict(registry_by_label or {})
    expansions: dict[str, GeneratorAlgebraicExpansion] = {}
    label_to_keys: dict[str, list[str]] = {}

    def _add(expansion: GeneratorAlgebraicExpansion) -> None:
        key = str(expansion.key)
        if key in expansions and expansions[key] != expansion:
            suffix = 2
            base = key
            while f"{base}#{suffix}" in expansions:
                suffix += 1
            key = f"{base}#{suffix}"
            expansion = GeneratorAlgebraicExpansion(
                key=key,
                label=expansion.label,
                generator_id=expansion.generator_id,
                terms=expansion.terms,
                support_qubits=expansion.support_qubits,
                exactness=expansion.exactness,
                source=expansion.source,
            )
        expansions[key] = expansion
        label_to_keys.setdefault(str(expansion.label), []).append(key)

    if pool is not None:
        for term in pool:
            label = str(getattr(term, "label", ""))
            meta = registry.get(label)
            if meta is not None and _serialized_terms_available(meta):
                _add(expansion_from_generator_metadata(meta, label=label, require_exact=True))
                continue
            if allow_polynomial_source and hasattr(term, "polynomial"):
                _add(expansion_from_ansatz_term(term, key=label))
                continue
            if meta is not None and not require_exact:
                _add(expansion_from_generator_metadata(meta, label=label, require_exact=False))
                continue
            if require_exact:
                raise AlgebraicMetadataError(
                    f"Pool term {label!r} is missing exact serialized metadata and polynomial fallback."
                )
    else:
        for label, meta in registry.items():
            _add(expansion_from_generator_metadata(meta, label=str(label), require_exact=bool(require_exact)))

    return AlgebraicMetadataIndex(
        expansions_by_key=expansions,
        label_to_keys={label: tuple(keys) for label, keys in label_to_keys.items()},
    )
