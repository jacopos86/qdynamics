"""Ordered insertion-plan words: enumeration, canonicalization, quotient.

Covers the selector specification's word-level requirements, including the
mandated counterexamples: the one-block ``Bba`` retention, the raw
``(n+r)!/n!`` count, and preservation of inequivalent no-deletion lifts.
"""

from __future__ import annotations

import random

import pytest

from pipelines.time_dynamics.ap_mclachlan.insertion_words import (
    INSERTED,
    SURVIVOR,
    WordToken,
    canonical_word,
    enumerate_raw_full_words,
    erase_tokens,
    inserted_tokens,
    quotient_insertion_plans,
    raw_full_word_count,
    survivor_tokens,
    tokens_commute_from_terms,
    word_keys,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _term(label: str, *components: tuple[str, float], nq: int = 4) -> AnsatzTerm:
    poly = PauliPolynomial("JW")
    for word, coeff in components:
        poly.add_term(PauliTerm(int(nq), ps=str(word), pc=float(coeff)))
    poly._reduce()
    return AnsatzTerm(label=str(label), polynomial=poly)


def _oracle(pairs_commute: dict[frozenset[str], bool]):
    """Token oracle from an explicit unordered-pair table (default: dependent)."""

    def commute(left: WordToken, right: WordToken) -> bool:
        if left.kind == SURVIVOR and right.kind == SURVIVOR:
            return False
        return pairs_commute.get(frozenset((left.key, right.key)), False)

    return commute


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, r",
    [(0, 1), (1, 1), (1, 2), (2, 2), (3, 2), (2, 3)],
)
def test_raw_full_word_count_matches_formula(n: int, r: int) -> None:
    survivors = survivor_tokens([f"s{i}" for i in range(n)])
    inserted = inserted_tokens([f"c{i}" for i in range(r)])
    words = list(enumerate_raw_full_words(survivors, inserted))
    assert len(words) == raw_full_word_count(n, r)
    assert len({tuple(word_keys(w)) for w in words}) == len(words)
    for word in words:
        surv = [t.key for t in word if t.kind == SURVIVOR]
        assert surv == [f"s{i}" for i in range(n)]  # survivor order preserved


def test_zero_inserted_yields_only_the_checkpoint_word() -> None:
    survivors = survivor_tokens(["s0", "s1"])
    assert list(enumerate_raw_full_words(survivors, ())) == [survivors]


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def test_canonical_word_never_reorders_survivors() -> None:
    survivors = survivor_tokens(["s0", "s1", "s2"])
    a = inserted_tokens(["a"])[0]
    word = (survivors[0], a, survivors[1], survivors[2])
    commute_all = _oracle({
        frozenset(("a", "s0")): True,
        frozenset(("a", "s1")): True,
        frozenset(("a", "s2")): True,
    })
    canon = canonical_word(word, commute_all)
    surv = [t.key for t in canon if t.kind == SURVIVOR]
    assert surv == ["s0", "s1", "s2"]
    # Fully commuting inserted token floats to its lexicographic slot but the
    # class is one word regardless of its raw position.
    other = (survivors[0], survivors[1], survivors[2], a)
    assert canonical_word(other, commute_all) == canon


def test_canonical_word_is_idempotent_and_class_invariant() -> None:
    survivors = survivor_tokens(["s0", "s1"])
    a, b = inserted_tokens(["a", "b"])
    table = {
        frozenset(("a", "s0")): True,
        frozenset(("a", "s1")): False,
        frozenset(("b", "s0")): False,
        frozenset(("b", "s1")): True,
        frozenset(("a", "b")): True,
    }
    commute = _oracle(table)
    rng = random.Random(7)
    words = list(enumerate_raw_full_words(survivors, (a, b)))
    canons = {canonical_word(w, commute) for w in words}
    for canon in canons:
        assert canonical_word(canon, commute) == canon  # idempotent
    # every word's canonical form is a member of its own multiset
    for w in words:
        assert sorted(word_keys(canonical_word(w, commute))) == sorted(word_keys(w))


# ---------------------------------------------------------------------------
# The mandated one-block counterexample
# ---------------------------------------------------------------------------


def test_one_block_counterexample_retains_bba() -> None:
    """[a,B]=0, [b,B]!=0, [a,b]!=0: the word Bba stays a distinct plan."""

    survivors = survivor_tokens(["B"])
    a, b = inserted_tokens(["a", "b"])
    commute = _oracle({
        frozenset(("a", "B")): True,
        frozenset(("b", "B")): False,
        frozenset(("a", "b")): False,
    })
    plans = quotient_insertion_plans(
        survivors=survivors,
        inserted=(a, b),
        removed_keys=(),
        tokens_commute=commute,
    )
    keys = {word_keys(plan.full_word) for plan in plans}
    # Classes: {abB}, {aBb, Bab}, {baB, bBa}, {Bba} -> four plans, each
    # retaining its lexicographically earliest raw member under the frozen
    # token order (survivor B before inserted a before inserted b).
    assert len(plans) == 4
    assert keys == {
        ("a", "b", "B"),
        ("a", "B", "b"),
        ("b", "a", "B"),
        ("B", "b", "a"),
    }
    # The mandated counterexample word Bba survives as its own plan.
    assert ("B", "b", "a") in {word_keys(p.canonical_full) for p in plans}
    assert ("B", "b", "a") in keys


def test_fully_commuting_inserted_pair_collapses_to_one_plan() -> None:
    survivors = survivor_tokens(["B"])
    a, b = inserted_tokens(["a", "b"])
    commute = _oracle({
        frozenset(("a", "B")): True,
        frozenset(("b", "B")): True,
        frozenset(("a", "b")): True,
    })
    plans = quotient_insertion_plans(
        survivors=survivors,
        inserted=(a, b),
        removed_keys=(),
        tokens_commute=commute,
    )
    assert len(plans) == 1
    assert raw_full_word_count(1, 2) == 6  # collapsed from six raw words


# ---------------------------------------------------------------------------
# Paired-word quotient and deletion erasure
# ---------------------------------------------------------------------------


def test_erase_tokens_requires_present_keys() -> None:
    survivors = survivor_tokens(["s0", "s1"])
    assert word_keys(erase_tokens(survivors, ("s0",))) == ("s1",)
    with pytest.raises(ValueError, match="absent from the word"):
        erase_tokens(survivors, ("nope",))


def test_inequivalent_no_deletion_lifts_are_preserved() -> None:
    """Plans whose reduced words coincide stay distinct when their full words
    differ: the deletion erases the survivor that distinguished them."""

    survivors = survivor_tokens(["B"])
    (a,) = inserted_tokens(["a"])
    commute = _oracle({frozenset(("a", "B")): False})  # aB and Ba inequivalent
    plans = quotient_insertion_plans(
        survivors=survivors,
        inserted=(a,),
        removed_keys=("B",),
        tokens_commute=commute,
    )
    # Both reduced words are just (a,), but the full words aB / Ba differ.
    assert len(plans) == 2
    assert {word_keys(p.reduced_word) for p in plans} == {("a",)}
    assert {word_keys(p.full_word) for p in plans} == {("a", "B"), ("B", "a")}


def test_equal_paired_words_collapse_even_across_reduced_difference() -> None:
    """A commuting crossing collapses full words; the reduced words then agree
    too, so exactly one plan survives."""

    survivors = survivor_tokens(["B"])
    (a,) = inserted_tokens(["a"])
    commute = _oracle({frozenset(("a", "B")): True})
    plans = quotient_insertion_plans(
        survivors=survivors,
        inserted=(a,),
        removed_keys=(),
        tokens_commute=commute,
    )
    assert len(plans) == 1


def test_retained_member_is_earliest_raw_word_and_output_is_deterministic() -> None:
    survivors = survivor_tokens(["s0", "s1"])
    a, b = inserted_tokens(["a", "b"])
    commute = _oracle({
        frozenset(("a", "s0")): True,
        frozenset(("a", "s1")): True,
        frozenset(("b", "s0")): True,
        frozenset(("b", "s1")): True,
        frozenset(("a", "b")): False,
    })
    first = quotient_insertion_plans(
        survivors=survivors, inserted=(a, b), removed_keys=(), tokens_commute=commute
    )
    second = quotient_insertion_plans(
        survivors=survivors, inserted=(a, b), removed_keys=(), tokens_commute=commute
    )
    assert first == second
    # a,b mutually dependent, both fully commuting with survivors: classes are
    # exactly the two inserted orders.
    assert len(first) == 2
    assert [p.inserted_keys for p in first] == [("a", "b"), ("b", "a")]


# ---------------------------------------------------------------------------
# Real-operator oracle
# ---------------------------------------------------------------------------


def test_tokens_commute_from_terms_matches_certificates_and_memoizes() -> None:
    X0 = _term("x0", ("eeex", 1.0))
    Z0 = _term("z0", ("eeez", 1.0))
    Z1 = _term("z1", ("eeze", 1.0))
    terms = {"x0": X0, "z0": Z0, "z1": Z1}
    commute = tokens_commute_from_terms(terms)
    s = survivor_tokens(["x0"])[0]
    (cz0,) = inserted_tokens(["z0"])
    cz1 = WordToken(kind=INSERTED, key="z1", sort_index=1)
    assert not commute(s, cz0)   # x,z overlap anticommutes
    assert commute(s, cz1)       # disjoint support
    assert not commute(s, cz0)   # memoized path returns identically
    # survivor-survivor is dependent regardless of algebra
    s2 = WordToken(kind=SURVIVOR, key="z1", sort_index=1)
    assert not commute(s, s2)


def test_singleton_quotient_matches_commutation_module_cuts() -> None:
    from pipelines.time_dynamics.ap_mclachlan.commutation import (
        singleton_insertion_cuts,
    )

    X0 = _term("x0", ("eeex", 1.0))
    X1 = _term("x1", ("eexe", 1.0))
    Z0 = _term("z0", ("eeez", 1.0))
    blocks = (X0, X1, Z0)
    terms = {"b0": X0, "b1": X1, "b2": Z0, "cand": Z0}
    survivors = survivor_tokens(["b0", "b1", "b2"])
    (cand,) = inserted_tokens(["cand"])
    commute = tokens_commute_from_terms(terms)
    plans = quotient_insertion_plans(
        survivors=survivors, inserted=(cand,), removed_keys=(), tokens_commute=commute
    )
    cuts = singleton_insertion_cuts(Z0, blocks)
    # One plan per retained cut; the retained full word inserts the candidate
    # after exactly `cut` survivors.
    positions = sorted(
        sum(1 for t in plan.full_word[: word_keys(plan.full_word).index("cand")] if t.kind == SURVIVOR)
        for plan in plans
    )
    assert tuple(positions) == cuts
