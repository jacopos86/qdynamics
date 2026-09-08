"""Lazy patch enumeration and an independent paired-word quotient oracle."""

from __future__ import annotations

from itertools import permutations

import pytest

from pipelines.time_dynamics.ap_mclachlan import insertion_words as words
from pipelines.time_dynamics.ap_mclachlan.patch_enumeration import (
    Patch,
    iter_permitted_patches,
)


def _lex(word):
    return tuple(token.lex_key for token in word)


def _canonical_by_adjacent_swaps(word, commute):
    """Small independent graph search; no production canonicalization."""
    visited = {word}
    pending = [word]
    while pending:
        current = pending.pop()
        for index in range(len(current) - 1):
            left, right = current[index : index + 2]
            if left.kind == right.kind == words.SURVIVOR or not commute(left, right):
                continue
            changed = current[:index] + (right, left) + current[index + 2 :]
            if changed not in visited:
                visited.add(changed)
                pending.append(changed)
    return min(visited, key=_lex)


def _brute_paired_quotient(survivors, inserted, removed, commute):
    classes = {}
    for raw in permutations(survivors + inserted):
        if tuple(token for token in raw if token.kind == words.SURVIVOR) != survivors:
            continue
        reduced = tuple(token for token in raw if token.key not in removed)
        canonical_full = _canonical_by_adjacent_swaps(raw, commute)
        canonical_reduced = _canonical_by_adjacent_swaps(reduced, commute)
        identity = canonical_full, canonical_reduced
        if identity not in classes or _lex(raw) < _lex(classes[identity][0]):
            classes[identity] = raw, reduced, canonical_full, canonical_reduced, removed
    return tuple(sorted(classes.values(), key=lambda row: _lex(row[0])))


@pytest.mark.parametrize("mask", range(8))
@pytest.mark.parametrize("removed", [(), ("s0",)])
def test_lazy_quotient_matches_independent_adjacent_swap_oracle(mask, removed) -> None:
    survivors = words.survivor_tokens(("s0",))
    inserted = words.inserted_tokens(("a", "b"))
    pairs = [("a", "s0"), ("b", "s0"), ("a", "b")]
    allowed = {frozenset(pair) for bit, pair in enumerate(pairs) if mask & (1 << bit)}

    def commute(left, right):
        return frozenset((left.key, right.key)) in allowed

    expected = _brute_paired_quotient(survivors, inserted, removed, commute)
    actual = tuple(
        words.iter_quotient_insertion_plans(
            survivors=survivors,
            inserted=inserted,
            removed_keys=removed,
            tokens_commute=commute,
        )
    )
    assert tuple(
        (p.full_word, p.reduced_word, p.canonical_full, p.canonical_reduced, p.removed_keys)
        for p in actual
    ) == expected
    assert words.quotient_insertion_plans(
        survivors=survivors,
        inserted=inserted,
        removed_keys=removed,
        tokens_commute=commute,
    ) == actual


@pytest.mark.parametrize("mask", [0, 3, 17, 31])
@pytest.mark.parametrize("removed", [(), ("s0",), ("s1",), ("s0", "s1")])
def test_lazy_quotient_preserves_two_survivor_paired_classes(mask, removed) -> None:
    survivors = words.survivor_tokens(("s0", "s1"))
    inserted = words.inserted_tokens(("a", "b"))
    pairs = [("a", "s0"), ("a", "s1"), ("b", "s0"), ("b", "s1"), ("a", "b")]
    allowed = {frozenset(pair) for bit, pair in enumerate(pairs) if mask & (1 << bit)}

    def commute(left, right):
        return frozenset((left.key, right.key)) in allowed

    actual = words.quotient_insertion_plans(
        survivors=survivors,
        inserted=inserted,
        removed_keys=removed,
        tokens_commute=commute,
    )
    expected = _brute_paired_quotient(survivors, inserted, removed, commute)

    assert tuple(
        (p.full_word, p.reduced_word, p.canonical_full, p.canonical_reduced, p.removed_keys)
        for p in actual
    ) == expected


def test_raw_words_are_globally_lexicographic_without_reordering_survivors() -> None:
    survivors = words.survivor_tokens(("s0", "s1"))
    inserted = words.inserted_tokens(("b", "a"))
    expected = sorted(
        (
            raw
            for raw in permutations(survivors + inserted)
            if tuple(t for t in raw if t.kind == words.SURVIVOR) == survivors
        ),
        key=_lex,
    )

    assert list(words.enumerate_raw_full_words(survivors, inserted)) == expected


def test_quotient_first_yield_does_not_request_a_future_raw_word(monkeypatch) -> None:
    survivors = words.survivor_tokens(("s0",))
    inserted = words.inserted_tokens(("a", "b"))
    raw = inserted + survivors

    def only_first_word(*args, **kwargs):
        yield raw
        pytest.fail("The first quotient result requested a future raw word.")

    monkeypatch.setattr(words, "enumerate_raw_full_words", only_first_word)
    stream = words.iter_quotient_insertion_plans(
        survivors=survivors,
        inserted=inserted,
        removed_keys=(),
        tokens_commute=lambda _left, _right: True,
    )

    assert next(stream).full_word == raw
    stream.close()


def test_raw_word_iterator_does_not_depend_on_python_recursion_depth() -> None:
    survivors = words.survivor_tokens(tuple(f"s{i}" for i in range(1100)))
    inserted = words.inserted_tokens(("a",))

    assert sum(1 for _ in words.enumerate_raw_full_words(survivors, inserted)) == 1101


def test_permission_filters_branch_before_its_pool_and_preserves_supplied_order() -> None:
    events = []

    def permission(removed):
        events.append(("permission", removed))
        return removed != (0,)

    def blocks(removed):
        events.append(("blocks", removed))
        return ((), ("b",), ("a",))

    actual = list(
        iter_permitted_patches(
            coordinate_keys=("s0", "s1"),
            deletion_sets=((0,), (1,), ()),
            insertion_blocks_for_deletion=blocks,
            singleton_cuts=lambda _removed, _atom: (2, 0),
            deletion_permission=permission,
        )
    )

    assert events == [
        ("permission", (0,)),
        ("permission", (1,)),
        ("blocks", (1,)),
        ("blocks", ()),
    ]
    assert actual == [
        Patch(removed, selection)
        for removed in ((1,), ())
        for selection in ((), (("b", 2),), (("b", 0),), (("a", 2),), (("a", 0),))
    ]


def test_first_patch_does_not_request_next_deletion_set_or_block() -> None:
    events = []

    def deletions():
        events.append("deletion")
        yield (1, 0)
        pytest.fail("First patch requested a successor deletion set.")

    def blocks(removed):
        events.append(("block", removed))
        yield ()
        pytest.fail("First patch requested a successor insertion block.")

    stream = iter_permitted_patches(
        coordinate_keys=("s0", "s1"),
        deletion_sets=deletions(),
        insertion_blocks_for_deletion=blocks,
        singleton_cuts=lambda _removed, _atom: (),
    )
    assert events == []
    assert next(stream) == Patch((0, 1), ())
    assert events == ["deletion", ("block", (0, 1))]
    stream.close()


def test_singleton_positions_are_requested_lazily() -> None:
    def cuts(_removed, _atom):
        yield 1
        pytest.fail("First singleton requested a successor position.")

    stream = iter_permitted_patches(
        coordinate_keys=("s0",),
        deletion_sets=((),),
        insertion_blocks_for_deletion=lambda _removed: (("a",),),
        singleton_cuts=cuts,
    )

    assert next(stream) == Patch((), (("a", 1),))
    stream.close()


def test_joint_patches_keep_full_word_cuts_and_inequivalent_deleted_lifts() -> None:
    actual = list(
        iter_permitted_patches(
            coordinate_keys=("s0",),
            deletion_sets=((0,),),
            insertion_blocks_for_deletion=lambda _removed: (("a", "b"),),
            singleton_cuts=lambda _removed, _atom: (),
            tokens_commute=lambda _left, _right: False,
        )
    )

    assert len(actual) == 6
    assert [patch.inserted_selection for patch in actual] == [
        (("a", 0), ("b", 0)),
        (("a", 0), ("b", 1)),
        (("b", 0), ("a", 0)),
        (("b", 0), ("a", 1)),
        (("a", 1), ("b", 1)),
        (("b", 1), ("a", 1)),
    ]
    assert all(patch.plan is not None for patch in actual)
    assert len({patch.plan.plan_id for patch in actual}) == 6
    assert {words.word_keys(patch.plan.reduced_word) for patch in actual} == {
        ("a", "b"), ("b", "a")
    }


def test_joint_patch_first_yield_is_lazy_through_word_quotient(monkeypatch) -> None:
    raw = words.inserted_tokens(("a", "b")) + words.survivor_tokens(("s0",))

    def only_first_word(*args, **kwargs):
        yield raw
        pytest.fail("First patch requested a successor raw word.")

    monkeypatch.setattr(words, "enumerate_raw_full_words", only_first_word)
    stream = iter_permitted_patches(
        coordinate_keys=("s0",),
        deletion_sets=((),),
        insertion_blocks_for_deletion=lambda _removed: (("a", "b"),),
        singleton_cuts=lambda _removed, _atom: (),
        tokens_commute=lambda _left, _right: True,
    )

    assert next(stream).inserted_selection == (("a", 0), ("b", 0))
    stream.close()


@pytest.mark.parametrize("deleted", [(-1,), (2,), (0, 0)])
def test_invalid_deletion_set_is_rejected_before_permission(deleted) -> None:
    def forbidden_permission(_removed):
        pytest.fail("Invalid deletion set reached permission.")

    stream = iter_permitted_patches(
        coordinate_keys=("s0", "s1"),
        deletion_sets=(deleted,),
        insertion_blocks_for_deletion=lambda _removed: ((),),
        singleton_cuts=lambda _removed, _atom: (),
        deletion_permission=forbidden_permission,
    )

    with pytest.raises(ValueError, match="distinct valid checkpoint indices"):
        next(stream)


def test_joint_insertion_requires_commutation_oracle() -> None:
    stream = iter_permitted_patches(
        coordinate_keys=(),
        deletion_sets=((),),
        insertion_blocks_for_deletion=lambda _removed: (("a", "b"),),
        singleton_cuts=lambda _removed, _atom: (),
    )

    with pytest.raises(ValueError, match="commutation oracle"):
        next(stream)
