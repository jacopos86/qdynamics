#!/usr/bin/env python3
"""Exact Pauli-expansion, support, and commutation metadata.

This module is the retained owner for representation-level algebra used by
static ADAPT geometry and recoverability pruning.  It deliberately contains no
candidate-lane assignment, lane quota, shortlist, or phase-routing policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


EXACTNESS_EXACT = "exact"
EXACTNESS_APPROX = "approx"

RELATION_FLAT_COMM = "flat_comm"
RELATION_CURV_NONCOMM = "curv_noncomm"
RELATION_DISJ_COMM = "disj_comm"
RELATION_APPROX_OR_UNKNOWN = "approx_or_unknown"


class AlgebraicMetadataError(ValueError):
    """Raised when exact expansion or commutation metadata is invalid."""


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
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "SerializedPauliExpansionTerm":
        nq = int(raw.get("nq", 0))
        word = normalize_pauli_word_exyz(raw.get("pauli_exyz", ""))
        if nq <= 0:
            raise AlgebraicMetadataError(
                f"Serialized Pauli term has invalid nq={nq!r}."
            )
        if len(word) != nq:
            raise AlgebraicMetadataError(
                f"Serialized Pauli word length {len(word)} does not match "
                f"nq={nq}: {word!r}."
            )
        if "coeff_re" not in raw or "coeff_im" not in raw:
            raise AlgebraicMetadataError(
                f"Serialized Pauli term {word!r} is missing "
                "coeff_re/coeff_im fields."
            )
        return cls(
            pauli_exyz=word,
            coeff_re=float(raw["coeff_re"]),
            coeff_im=float(raw["coeff_im"]),
            nq=int(nq),
        )


@dataclass(frozen=True)
class GeneratorAlgebraicExpansion:
    """Exact or explicitly approximate expansion for one generator."""

    key: str
    label: str
    generator_id: str | None
    terms: tuple[SerializedPauliExpansionTerm, ...]
    support_qubits: tuple[int, ...]
    exactness: str = EXACTNESS_EXACT
    source: str = "registry_serialized_terms"


@dataclass(frozen=True)
class AlgebraicPairMetadata:
    """Exact support and commutation relation for two generator expansions."""

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


@dataclass
class AlgebraicMetadataIndex:
    """Lazy exact-expansion and pair-commutation index."""

    expansions_by_key: Mapping[str, GeneratorAlgebraicExpansion]
    label_to_keys: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    pair_cache: dict[
        tuple[str, str, float],
        AlgebraicPairMetadata,
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expansions = {
            str(key): value for key, value in self.expansions_by_key.items()
        }
        labels: dict[str, list[str]] = {}
        for key, expansion in expansions.items():
            labels.setdefault(str(expansion.label), []).append(str(key))
        for label, keys in self.label_to_keys.items():
            labels.setdefault(str(label), [])
            for key in keys:
                if str(key) not in labels[str(label)]:
                    labels[str(label)].append(str(key))
        self.expansions_by_key = expansions
        self.label_to_keys = {
            label: tuple(keys) for label, keys in labels.items()
        }

    def resolve_key(self, key_or_label: str) -> str:
        raw = str(key_or_label)
        if raw in self.expansions_by_key:
            return raw
        keys = tuple(self.label_to_keys.get(raw, ()))
        if len(keys) == 1:
            return str(keys[0])
        if len(keys) > 1:
            raise AlgebraicMetadataError(
                f"Algebraic label {raw!r} is ambiguous; use one of keys "
                f"{keys!r}."
            )
        raise AlgebraicMetadataError(
            f"Unknown algebraic expansion key or label: {raw!r}."
        )

    def pair(
        self,
        lhs_key: str,
        rhs_key: str,
        *,
        coefficient_tol: float = 1.0e-12,
    ) -> AlgebraicPairMetadata:
        lhs = self.resolve_key(lhs_key)
        rhs = self.resolve_key(rhs_key)
        cache_key = (lhs, rhs, float(coefficient_tol))
        cached = self.pair_cache.get(cache_key)
        if cached is not None:
            return cached
        metadata = build_pair_metadata(
            self.expansions_by_key[lhs],
            self.expansions_by_key[rhs],
            coefficient_tol=float(coefficient_tol),
        )
        self.pair_cache[cache_key] = metadata
        return metadata


def normalize_pauli_word_exyz(
    raw: Any,
    *,
    require_exyz: bool = True,
) -> str:
    """Normalize a Pauli word to lowercase repo-native e/x/y/z symbols."""

    word = str(raw).strip().lower()
    if require_exyz:
        invalid = sorted(
            {character for character in word if character not in {"e", "x", "y", "z"}}
        )
        if invalid:
            raise AlgebraicMetadataError(
                f"Pauli word {raw!r} contains non e/x/y/z symbols: "
                f"{invalid!r}."
            )
    return word


def support_qubits_from_pauli_word(
    word: str,
    *,
    nq: int | None = None,
) -> tuple[int, ...]:
    """Return support qubits using qubit 0 as the rightmost character."""

    normalized = normalize_pauli_word_exyz(word)
    n_qubits = len(normalized) if nq is None else int(nq)
    if len(normalized) != n_qubits:
        raise AlgebraicMetadataError(
            f"Pauli word length {len(normalized)} does not match "
            f"nq={n_qubits}: {normalized!r}."
        )
    return tuple(
        sorted(
            int(n_qubits - 1 - index)
            for index, character in enumerate(normalized)
            if character != "e"
        )
    )


def pauli_words_commute(lhs: str, rhs: str) -> bool:
    """Return the exact parity-commutation result for two Pauli words."""

    left = normalize_pauli_word_exyz(lhs)
    right = normalize_pauli_word_exyz(rhs)
    if len(left) != len(right):
        raise AlgebraicMetadataError(
            f"Pauli words must have equal length, got {len(left)} and "
            f"{len(right)}."
        )
    anticommutes = 0
    for left_character, right_character in zip(left, right):
        if (
            left_character == "e"
            or right_character == "e"
            or left_character == right_character
        ):
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
            f"Pauli words must have equal length, got {len(left)} and "
            f"{len(right)}."
        )
    output: list[str] = []
    phase = complex(1.0)
    for left_character, right_character in zip(left, right):
        character, local_phase = _PAULI_PRODUCT[
            (left_character, right_character)
        ]
        output.append(character)
        phase *= complex(local_phase)
    return "".join(output), phase


def _terms_from_polynomial(
    polynomial: Any,
) -> tuple[SerializedPauliExpansionTerm, ...]:
    if polynomial is None or not hasattr(polynomial, "return_polynomial"):
        raise AlgebraicMetadataError(
            "Ansatz term is missing a PauliPolynomial-like polynomial."
        )
    output: list[SerializedPauliExpansionTerm] = []
    for term in polynomial.return_polynomial():
        coefficient = complex(term.p_coeff)
        output.append(
            SerializedPauliExpansionTerm(
                pauli_exyz=normalize_pauli_word_exyz(term.pw2strng()),
                coeff_re=float(coefficient.real),
                coeff_im=float(coefficient.imag),
                nq=int(term.nqubit()),
            )
        )
    if not output:
        raise AlgebraicMetadataError(
            "Ansatz polynomial has no Pauli terms for exact metadata."
        )
    return tuple(output)


def _support_from_terms(
    terms: Sequence[SerializedPauliExpansionTerm],
) -> tuple[int, ...]:
    support: set[int] = set()
    for term in terms:
        support.update(
            support_qubits_from_pauli_word(
                term.pauli_exyz,
                nq=int(term.nq),
            )
        )
    return tuple(sorted(int(qubit) for qubit in support))


def _meta_get(meta: Any, key: str, default: Any = None) -> Any:
    if isinstance(meta, Mapping):
        return meta.get(key, default)
    return getattr(meta, key, default)


def _is_nonstring_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _serialized_terms_available(meta: Any) -> bool:
    compile_metadata = _meta_get(meta, "compile_metadata", {})
    if not isinstance(compile_metadata, Mapping):
        return False
    raw_terms = compile_metadata.get("serialized_terms_exyz")
    if raw_terms is None:
        return False
    if not _is_nonstring_sequence(raw_terms):
        raise AlgebraicMetadataError(
            "compile_metadata.serialized_terms_exyz must be a sequence "
            "of term mappings."
        )
    return bool(len(raw_terms) > 0)


def expansion_from_generator_metadata(
    meta: Any,
    *,
    key: str | None = None,
    label: str | None = None,
    require_exact: bool = True,
) -> GeneratorAlgebraicExpansion:
    """Build an expansion from serialized generator metadata."""

    compile_metadata = _meta_get(meta, "compile_metadata", {})
    if not isinstance(compile_metadata, Mapping):
        compile_metadata = {}
    raw_terms = compile_metadata.get("serialized_terms_exyz")
    candidate_label = str(
        label or _meta_get(meta, "candidate_label", key or "")
    )
    generator_id_raw = _meta_get(meta, "generator_id", None)
    generator_id = (
        str(generator_id_raw) if generator_id_raw is not None else None
    )
    expansion_key = str(key or generator_id or candidate_label)
    if not _is_nonstring_sequence(raw_terms) or len(raw_terms) == 0:
        if require_exact:
            raise AlgebraicMetadataError(
                f"Generator {candidate_label!r} is missing exact "
                "compile_metadata.serialized_terms_exyz."
            )
        support_raw = _meta_get(meta, "support_qubits", ())
        support = (
            tuple(sorted(int(qubit) for qubit in support_raw))
            if _is_nonstring_sequence(support_raw)
            else ()
        )
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
    for raw_term in raw_terms:
        if not isinstance(raw_term, Mapping):
            raise AlgebraicMetadataError(
                f"Generator {candidate_label!r} has a non-mapping "
                f"serialized Pauli term: {raw_term!r}."
            )
        terms.append(SerializedPauliExpansionTerm.from_mapping(raw_term))
    return GeneratorAlgebraicExpansion(
        key=expansion_key,
        label=candidate_label,
        generator_id=generator_id,
        terms=tuple(terms),
        support_qubits=_support_from_terms(terms),
        exactness=EXACTNESS_EXACT,
        source="registry_serialized_terms",
    )


def expansion_from_ansatz_term(
    term: Any,
    *,
    key: str | None = None,
) -> GeneratorAlgebraicExpansion:
    """Extract an exact expansion from an ``AnsatzTerm.polynomial``."""

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
    accumulated: dict[str, complex] = {}
    for lhs in lhs_terms:
        for rhs in rhs_terms:
            if int(lhs.nq) != int(rhs.nq):
                raise AlgebraicMetadataError(
                    "Cannot commute expansions with inconsistent nq values: "
                    f"{lhs.nq} vs {rhs.nq}."
                )
            lhs_rhs_word, lhs_rhs_phase = multiply_pauli_words(
                lhs.pauli_exyz,
                rhs.pauli_exyz,
            )
            rhs_lhs_word, rhs_lhs_phase = multiply_pauli_words(
                rhs.pauli_exyz,
                lhs.pauli_exyz,
            )
            coefficient = lhs.coeff * rhs.coeff
            accumulated[lhs_rhs_word] = (
                accumulated.get(lhs_rhs_word, 0.0j)
                + coefficient * lhs_rhs_phase
            )
            accumulated[rhs_lhs_word] = (
                accumulated.get(rhs_lhs_word, 0.0j)
                - coefficient * rhs_lhs_phase
            )
    return float(sum(abs(value) for value in accumulated.values()))


def exact_expansions_commute(
    lhs: GeneratorAlgebraicExpansion,
    rhs: GeneratorAlgebraicExpansion,
    *,
    coefficient_tol: float = 1.0e-12,
) -> tuple[bool, float]:
    """Check the full-polynomial commutator, including term cancellation."""

    if (
        lhs.exactness != EXACTNESS_EXACT
        or rhs.exactness != EXACTNESS_EXACT
    ):
        raise AlgebraicMetadataError(
            "Exact commutation requires exact expansions for both generators."
        )
    if not lhs.terms or not rhs.terms:
        raise AlgebraicMetadataError(
            "Exact commutation requires non-empty Pauli expansions."
        )
    l1_norm = _commutator_l1_norm_from_terms(lhs.terms, rhs.terms)
    return bool(l1_norm <= float(coefficient_tol)), float(l1_norm)


def build_pair_metadata(
    lhs: GeneratorAlgebraicExpansion,
    rhs: GeneratorAlgebraicExpansion,
    *,
    coefficient_tol: float = 1.0e-12,
) -> AlgebraicPairMetadata:
    """Build exact support and commutation metadata for two expansions."""

    overlap_qubits = tuple(
        sorted(set(lhs.support_qubits).intersection(rhs.support_qubits))
    )
    support_overlap = bool(overlap_qubits)
    if (
        lhs.exactness != EXACTNESS_EXACT
        or rhs.exactness != EXACTNESS_EXACT
    ):
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
    commutes, l1_norm = exact_expansions_commute(
        lhs,
        rhs,
        coefficient_tol=float(coefficient_tol),
    )
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


def build_exact_expansion_index(
    *,
    pool: Sequence[Any] | None = None,
    registry_by_label: Mapping[str, Any] | None = None,
    require_exact: bool = True,
    allow_polynomial_source: bool = True,
) -> AlgebraicMetadataIndex:
    """Build an exact expansion index from metadata and/or ansatz terms."""

    registry = dict(registry_by_label or {})
    expansions: dict[str, GeneratorAlgebraicExpansion] = {}
    label_to_keys: dict[str, list[str]] = {}

    def add(expansion: GeneratorAlgebraicExpansion) -> None:
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
            metadata = registry.get(label)
            if metadata is not None and _serialized_terms_available(metadata):
                add(
                    expansion_from_generator_metadata(
                        metadata,
                        label=label,
                        require_exact=True,
                    )
                )
                continue
            if allow_polynomial_source and hasattr(term, "polynomial"):
                add(expansion_from_ansatz_term(term, key=label))
                continue
            if metadata is not None and not require_exact:
                add(
                    expansion_from_generator_metadata(
                        metadata,
                        label=label,
                        require_exact=False,
                    )
                )
                continue
            if require_exact:
                raise AlgebraicMetadataError(
                    f"Pool term {label!r} is missing exact serialized "
                    "metadata and polynomial fallback."
                )
    else:
        for label, metadata in registry.items():
            add(
                expansion_from_generator_metadata(
                    metadata,
                    label=str(label),
                    require_exact=bool(require_exact),
                )
            )

    return AlgebraicMetadataIndex(
        expansions_by_key=expansions,
        label_to_keys={
            label: tuple(keys) for label, keys in label_to_keys.items()
        },
    )


def ensure_exact_expansion_in_index(
    index: AlgebraicMetadataIndex,
    term: Any,
    registry_by_label: Mapping[str, Any],
    *,
    allow_polynomial_source: bool = True,
) -> bool:
    """Ensure one term has exact expansion metadata in ``index``.

    Runtime-split and newly admitted terms may not have been present when the
    initial pool index was constructed.  This helper extends the same mutable
    exact-expansion index without attaching any lane, shortlist, or routing
    meaning to the commutation record.
    """

    label = str(getattr(term, "label", ""))
    try:
        index.resolve_key(label)
        return True
    except AlgebraicMetadataError:
        pass

    single = build_exact_expansion_index(
        pool=[term],
        registry_by_label=registry_by_label,
        require_exact=True,
        allow_polynomial_source=bool(allow_polynomial_source),
    )
    for key, expansion in single.expansions_by_key.items():
        index.expansions_by_key[str(key)] = expansion
    for key_label, keys in single.label_to_keys.items():
        merged = list(index.label_to_keys.get(str(key_label), ()))
        for key in keys:
            if str(key) not in merged:
                merged.append(str(key))
        index.label_to_keys[str(key_label)] = tuple(merged)
    index.resolve_key(label)
    return True


__all__ = (
    "EXACTNESS_APPROX",
    "EXACTNESS_EXACT",
    "RELATION_APPROX_OR_UNKNOWN",
    "RELATION_CURV_NONCOMM",
    "RELATION_DISJ_COMM",
    "RELATION_FLAT_COMM",
    "AlgebraicMetadataError",
    "AlgebraicMetadataIndex",
    "AlgebraicPairMetadata",
    "GeneratorAlgebraicExpansion",
    "SerializedPauliExpansionTerm",
    "build_exact_expansion_index",
    "build_pair_metadata",
    "ensure_exact_expansion_in_index",
    "exact_expansions_commute",
    "expansion_from_ansatz_term",
    "expansion_from_generator_metadata",
    "multiply_pauli_words",
    "normalize_pauli_word_exyz",
    "pauli_words_commute",
    "support_qubits_from_pauli_word",
)
