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
from itertools import permutations
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

    inserted = tuple(inserted)
    n, r = len(survivors), len(inserted)
    if r == 0:
        yield tuple(survivors)
        return

    seen_orders: set[tuple[str, ...]] = set()
    for order in permutations(range(r)):
        order_key = tuple(inserted[i].key for i in order)
        if order_key in seen_orders:
            continue
        seen_orders.add(order_key)
        ordered = tuple(inserted[i] for i in order)
        # Choose the r slots (with repetition over n+1 gaps, ordered) by
        # enumerating positions of inserted tokens among n + r total slots.
        for positions in _combinations(n + r, r):
            word: list[WordToken] = []
            survivor_iter = iter(survivors)
            inserted_iter = iter(ordered)
            position_set = set(positions)
            for slot in range(n + r):
                word.append(
                    next(inserted_iter) if slot in position_set else next(survivor_iter)
                )
            yield tuple(word)


def _combinations(total: int, choose: int) -> Iterator[tuple[int, ...]]:
    from itertools import combinations

    yield from combinations(range(total), choose)


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


def quotient_insertion_plans(
    *,
    survivors: Word,
    inserted: Sequence[WordToken],
    removed_keys: Sequence[str],
    tokens_commute: TokensCommute,
) -> tuple[InsertionPlan, ...]:
    """Enumerate raw full words and retain one plan per paired-word class.

    ``removed_keys`` are erased from every full word to form its reduced word;
    two raw plans are equivalent only when *both* canonical words match, so an
    inequivalent no-deletion lift is preserved even when the reduced words
    coincide.  The retained member of each class is the lexicographically
    earliest raw word under the frozen token order, and the output is sorted
    by that same order for deterministic downstream scoring.
    """

    removed = tuple(str(key) for key in removed_keys)
    survivor_keys = {token.key for token in survivors}
    unknown = set(removed) - survivor_keys
    if unknown:
        raise ValueError(
            f"removed_keys must identify survivors of the checkpoint word; "
            f"unknown: {sorted(unknown)}."
        )

    def raw_order(word: Word) -> tuple[tuple[int, int, str], ...]:
        return tuple(token.lex_key for token in word)

    retained: dict[tuple, Word] = {}
    for raw in enumerate_raw_full_words(survivors, inserted):
        reduced = erase_tokens(raw, removed)
        identity = (
            word_keys(canonical_word(raw, tokens_commute)),
            word_keys(canonical_word(reduced, tokens_commute)),
        )
        best = retained.get(identity)
        if best is None or raw_order(raw) < raw_order(best):
            retained[identity] = raw

    plans = tuple(
        InsertionPlan(
            full_word=raw,
            reduced_word=erase_tokens(raw, removed),
            canonical_full=canonical_word(raw, tokens_commute),
            canonical_reduced=canonical_word(
                erase_tokens(raw, removed), tokens_commute
            ),
            removed_keys=removed,
        )
        for raw in sorted(retained.values(), key=raw_order)
    )
    return plans


__all__ = [
    "INSERTED",
    "SURVIVOR",
    "InsertionPlan",
    "Word",
    "WordToken",
    "canonical_word",
    "enumerate_raw_full_words",
    "erase_tokens",
    "inserted_tokens",
    "quotient_insertion_plans",
    "raw_full_word_count",
    "survivor_tokens",
    "tokens_commute_from_terms",
    "word_keys",
]
