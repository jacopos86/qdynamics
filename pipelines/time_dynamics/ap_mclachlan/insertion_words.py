"""Ordered insertion-plan words and their commutation quotient.

The deletion-conditioned exchange selector represents every structural
candidate as a typed pair of circuit words:

* the **full word** ``W`` — an interleaving of the frozen checkpoint sequence
  ``J_k = (j_1..j_n)`` with the inserted child occurrences, preserving the
  relative order of the original tokens; and
* the **reduced word** ``W_D = erase_D(W)`` — the same word with the deleted
  coordinates erased, which is the layout a committing patch materializes.

Two raw plans are equivalent when both their full words and their reduced
words canonicalize to the same representatives under adjacent swaps of
certifiably commuting tokens.  Survivor--survivor order is never swapped, so
the reachable words of one class are exactly the linearizations of the
dependence partial order in which every dependent pair keeps its original
relative order (a trace-monoid equivalence over unique token occurrences).
The canonical form is therefore the lexicographically least topological
linearization of that dependence DAG, and the retained member of each class
is the lexicographically earliest *raw* pair, per the selector specification.

Commutation between tokens is delegated to
:mod:`pipelines.time_dynamics.ap_mclachlan.commutation` (Paper-I semantics);
this module never inspects operators itself.  Certificates are memoized per
unordered key pair because the quotient evaluates each pair O(word length)
times.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterator, Mapping, Sequence

from pipelines.time_dynamics.ap_mclachlan.commutation import (
    DEFAULT_COEFFICIENT_TOLERANCE,
    generator_blocks_commute,
)


SURVIVOR = "survivor"
INSERTED = "inserted"


@dataclass(frozen=True, slots=True)
class WordToken:
    """One occurrence in a circuit word.

    ``key`` is the immutable identity (frozen coordinate id for survivors,
    child occurrence id for inserted tokens).  ``sort_index`` fixes the frozen
    total order used for lexicographic comparison: survivors carry their
    original position, inserted tokens their frozen candidate-order index.

    Inserted tokens order *before* survivors.  This is load-bearing, not
    cosmetic: within one commutation class an inserted token can slide across
    every commuting survivor, and the lexicographically least member must be
    the one that places it at the earliest reachable cut, so that the
    word-level quotient retains exactly the Paper-I earliest-representative
    insertion positions in the singleton specialization.
    """

    kind: str
    key: str
    sort_index: int

    def __post_init__(self) -> None:
        if self.kind not in (SURVIVOR, INSERTED):
            raise ValueError(f"Unsupported token kind: {self.kind!r}.")

    @property
    def lex_key(self) -> tuple[int, int, str]:
        return (0 if self.kind == INSERTED else 1, int(self.sort_index), self.key)


Word = tuple[WordToken, ...]
TokensCommute = Callable[[WordToken, WordToken], bool]


def survivor_tokens(keys: Sequence[str]) -> Word:
    """Frozen checkpoint word ``J_k`` as survivor tokens in given order."""

    out = tuple(
        WordToken(kind=SURVIVOR, key=str(key), sort_index=int(index))
        for index, key in enumerate(keys)
    )
    if len({token.key for token in out}) != len(out):
        raise ValueError("survivor keys must be unique.")
    return out


def inserted_tokens(keys: Sequence[str]) -> tuple[WordToken, ...]:
    """Candidate child occurrences in frozen candidate order."""

    out = tuple(
        WordToken(kind=INSERTED, key=str(key), sort_index=int(index))
        for index, key in enumerate(keys)
    )
    if len({token.key for token in out}) != len(out):
        raise ValueError("inserted occurrence keys must be unique.")
    return out


def tokens_commute_from_terms(
    terms_by_key: Mapping[str, Any],
    *,
    coefficient_tolerance: float = DEFAULT_COEFFICIENT_TOLERANCE,
) -> TokensCommute:
    """Build the token-level commutation oracle from generator terms.

    Survivor--survivor pairs are always dependent (their order is frozen), so
    the oracle is only consulted for pairs with at least one inserted token;
    it still guards the rule itself for safety.  Certificates are memoized on
    the unordered key pair.
    """

    cache: dict[tuple[str, str], bool] = {}

    def commute(left: WordToken, right: WordToken) -> bool:
        if left.kind == SURVIVOR and right.kind == SURVIVOR:
            return False
        pair = (min(left.key, right.key), max(left.key, right.key))
        cached = cache.get(pair)
        if cached is None:
            cached = generator_blocks_commute(
                terms_by_key[left.key],
                terms_by_key[right.key],
                coefficient_tolerance=coefficient_tolerance,
            )
            cache[pair] = cached
        return cached

    return commute


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def canonical_word(word: Word, tokens_commute: TokensCommute) -> Word:
    """Lexicographically least word reachable by certified adjacent swaps.

    Tokens ``x`` before ``y`` are *dependent* when their swap is not allowed
    (both survivors, or the commutation certificate fails); the equivalence
    class of ``word`` is every linearization of the dependence DAG, and the
    canonical form is its lexicographically least topological order under
    :attr:`WordToken.lex_key`.
    """

    size = len(word)
    if size <= 1:
        return tuple(word)

    dependent_predecessors: list[set[int]] = [set() for _ in range(size)]
    for later in range(size):
        for earlier in range(later):
            left, right = word[earlier], word[later]
            if left.kind == SURVIVOR and right.kind == SURVIVOR:
                dependent_predecessors[later].add(earlier)
            elif not tokens_commute(left, right):
                dependent_predecessors[later].add(earlier)

    emitted: set[int] = set()
    out: list[WordToken] = []
    remaining = set(range(size))
    while remaining:
        ready = [
            index
            for index in remaining
            if dependent_predecessors[index] <= emitted
        ]
        chosen = min(ready, key=lambda index: word[index].lex_key)
        emitted.add(chosen)
        remaining.remove(chosen)
        out.append(word[chosen])
    return tuple(out)


def erase_tokens(word: Word, removed_keys: Sequence[str]) -> Word:
    """Return ``erase_D(word)``: the word with the removed identities deleted."""

    removed = {str(key) for key in removed_keys}
    missing = removed - {token.key for token in word}
    if missing:
        raise ValueError(f"Cannot erase keys absent from the word: {sorted(missing)}.")
    return tuple(token for token in word if token.key not in removed)


def word_keys(word: Word) -> tuple[str, ...]:
    return tuple(token.key for token in word)


# ---------------------------------------------------------------------------
# Enumeration and quotient
# ---------------------------------------------------------------------------


def raw_full_word_count(survivor_count: int, inserted_count: int) -> int:
    """``(n + r)! / n!`` raw interleavings before quotienting."""

    n = int(survivor_count)
    r = int(inserted_count)
    if n < 0 or r < 0:
        raise ValueError("counts must be non-negative.")
    return math.factorial(n + r) // math.factorial(n)


def enumerate_raw_full_words(
    survivors: Word,
    inserted: Sequence[WordToken],
) -> Iterator[Word]:
    """Yield every interleaving of ``survivors`` (order fixed) with the
    inserted occurrences (any order), in deterministic lexicographic order of
    the emitted words."""

    survivors = tuple(survivors)
    inserted = tuple(sorted(inserted, key=lambda token: token.lex_key))
    # Depth-first traversal of legal next tokens gives global lexicographic
    # order without collecting permutations or words. The explicit stack also
    # avoids a recursion limit tied to the checkpoint support size.
    pending = [(0, inserted, ())]
    while pending:
        survivor_index, remaining, prefix = pending.pop()
        if not remaining:
            yield prefix + survivors[survivor_index:]
            continue
        # Inserted tokens precede survivors in the frozen token order. Push
        # later choices first so the stack visits the earliest choice first.
        if survivor_index < len(survivors):
            pending.append(
                (survivor_index + 1, remaining, prefix + (survivors[survivor_index],))
            )
        for index in range(len(remaining) - 1, -1, -1):
            pending.append(
                (
                    survivor_index,
                    remaining[:index] + remaining[index + 1 :],
                    prefix + (remaining[index],),
                )
            )


@dataclass(frozen=True, slots=True)
class InsertionPlan:
    """One retained ``(W, W_D)`` insertion plan.

    ``full_word``/``reduced_word`` are the retained raw members (earliest in
    the class); the canonical pair is the class identity used for cache keys
    and equality.  ``inserted_keys`` preserves the order the plan inserts its
    children in the full word.
    """

    full_word: Word
    reduced_word: Word
    canonical_full: Word
    canonical_reduced: Word
    removed_keys: tuple[str, ...]

    @property
    def inserted_keys(self) -> tuple[str, ...]:
        return tuple(t.key for t in self.full_word if t.kind == INSERTED)

    @property
    def plan_id(self) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Stable identity: canonical word keys plus the deletion set."""

        return (
            word_keys(self.canonical_full),
            word_keys(self.canonical_reduced),
            tuple(sorted(self.removed_keys)),
        )


def iter_quotient_insertion_plans(
    *,
    survivors: Word,
    inserted: Sequence[WordToken],
    removed_keys: Sequence[str],
    tokens_commute: TokensCommute,
) -> Iterator[InsertionPlan]:
    """Yield the earliest raw member of each paired-word class lazily.

    ``removed_keys`` are erased from every full word to form its reduced word;
    two raw plans are equivalent only when *both* canonical words match, so an
    inequivalent no-deletion lift is preserved even when the reduced words
    coincide.  The retained member of each class is the lexicographically
    earliest raw word under the frozen token order. Raw words arrive in that
    order, so a class's first appearance is final and can be yielded at once.
    Only identities of already emitted classes are retained; no future word
    is visited merely to return the current plan.
    """

    removed = tuple(str(key) for key in removed_keys)
    survivor_keys = {token.key for token in survivors}
    unknown = set(removed) - survivor_keys
    if unknown:
        raise ValueError(
            f"removed_keys must identify survivors of the checkpoint word; "
            f"unknown: {sorted(unknown)}."
        )

    seen: set[tuple] = set()
    for raw in enumerate_raw_full_words(survivors, inserted):
        reduced = erase_tokens(raw, removed)
        canonical_full = canonical_word(raw, tokens_commute)
        canonical_reduced = canonical_word(reduced, tokens_commute)
        identity = (
            word_keys(canonical_full),
            word_keys(canonical_reduced),
        )
        if identity in seen:
            continue
        seen.add(identity)
        yield InsertionPlan(
            full_word=raw,
            reduced_word=reduced,
            canonical_full=canonical_full,
            canonical_reduced=canonical_reduced,
            removed_keys=removed,
        )


def quotient_insertion_plans(
    *,
    survivors: Word,
    inserted: Sequence[WordToken],
    removed_keys: Sequence[str],
    tokens_commute: TokensCommute,
) -> tuple[InsertionPlan, ...]:
    """Return the complete paired-word quotient in frozen lexicographic order."""

    return tuple(
        iter_quotient_insertion_plans(
            survivors=survivors,
            inserted=inserted,
            removed_keys=removed_keys,
            tokens_commute=tokens_commute,
        )
    )


def full_word_insertions(word: Word) -> tuple[tuple[str, int], ...]:
    """Inserted occurrences at their raw full-word cuts, in word order.

    A cut counts preceding checkpoint tokens, including ones later deleted.
    Mapping these positions to tangent-cache representatives belongs to the
    geometry adapter; materialization uses the plan's reduced word.
    """

    selection: list[tuple[str, int]] = []
    survivor_count = 0
    for token in word:
        if token.kind == SURVIVOR:
            survivor_count += 1
        else:
            selection.append((token.key, survivor_count))
    return tuple(selection)


__all__ = [
    "INSERTED",
    "SURVIVOR",
    "InsertionPlan",
    "Word",
    "WordToken",
    "canonical_word",
    "enumerate_raw_full_words",
    "erase_tokens",
    "full_word_insertions",
    "inserted_tokens",
    "iter_quotient_insertion_plans",
    "quotient_insertion_plans",
    "raw_full_word_count",
    "survivor_tokens",
    "tokens_commute_from_terms",
    "word_keys",
]
