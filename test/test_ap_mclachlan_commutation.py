"""Commutation certificates and insertion cuts for the exchange selector.

The load-bearing property is cross-lane parity: the Paper-II cut reduction
must agree with Paper I's insertion-position reduction on identical inputs,
because the spec requires "the same retained-Pauli-component cutoff and
commutation rule as Paper I".
"""

from __future__ import annotations

import pytest

from pipelines.static_adapt.adapt_pipeline import (
    _commutation_reduced_insertion_position_plan,
    _termwise_generators_commute_for_insertion,
)
from pipelines.time_dynamics.ap_mclachlan.commutation import (
    generator_blocks_commute,
    reduce_insertion_positions,
    singleton_insertion_cuts,
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


# Single-word generators on 4 qubits (q3 q2 q1 q0 ordering).
X0 = _term("x0", ("eeex", 1.0))
Z0 = _term("z0", ("eeez", 1.0))
X1 = _term("x1", ("eexe", 1.0))
Z1 = _term("z1", ("eeze", 1.0))
XX01 = _term("xx01", ("eexx", 0.5))
# Macro generators with several Pauli components.
MACRO_A = _term("macro_a", ("eexx", 0.5), ("eeyy", 0.5))
MACRO_B = _term("macro_b", ("xxee", 0.25), ("yyee", 0.25))
MACRO_MIXED = _term("macro_mixed", ("eexx", 0.5), ("zeee", 0.3))
TINY = _term("tiny", ("eeex", 1.0e-15))  # below coefficient tolerance


# ---------------------------------------------------------------------------
# Certificate semantics
# ---------------------------------------------------------------------------


def test_certificate_matches_word_level_algebra() -> None:
    assert generator_blocks_commute(X0, X0)
    assert not generator_blocks_commute(X0, Z0)          # overlap, anticommute
    assert generator_blocks_commute(X0, Z1)              # disjoint support
    assert generator_blocks_commute(MACRO_A, MACRO_B)    # disjoint macros
    # Termwise rule: every component pair must commute.  eexx commutes with
    # eexx and with zeee, but eeyy anticommutes with... nothing here; check a
    # genuinely mixed failure: MACRO_MIXED's zeee commutes with MACRO_B's xxee?
    # z on q3 vs x on q3 anticommute -> certificate must fail.
    assert not generator_blocks_commute(MACRO_MIXED, MACRO_B)


def test_certificate_is_conservative_for_empty_or_tiny_blocks() -> None:
    assert not generator_blocks_commute(TINY, X0)
    assert not generator_blocks_commute(X0, TINY)


@pytest.mark.parametrize(
    "left, right",
    [(X0, Z0), (X0, Z1), (MACRO_A, MACRO_B), (MACRO_MIXED, MACRO_B), (XX01, Z0)],
)
def test_certificate_parity_with_paper_i(left, right) -> None:
    assert generator_blocks_commute(left, right) == (
        _termwise_generators_commute_for_insertion(left, right)
    )
    # Symmetric, both implementations.
    assert generator_blocks_commute(right, left) == (
        _termwise_generators_commute_for_insertion(right, left)
    )


# ---------------------------------------------------------------------------
# Singleton insertion cuts
# ---------------------------------------------------------------------------


def test_singleton_cuts_are_zero_union_uncrossable_blocks() -> None:
    # Candidate Z0 against blocks [X0, X1, Z0]:
    # crosses X0? z/x on q0 anticommute -> no. X1 disjoint -> yes. Z0 same -> yes.
    assert singleton_insertion_cuts(Z0, (X0, X1, Z0)) == (0, 1)
    # Candidate X0 against [Z0, Z0, Z0]: nothing crossable.
    assert singleton_insertion_cuts(X0, (Z0, Z0, Z0)) == (0, 1, 2, 3)
    # Fully commuting candidate collapses every cut to 0.
    assert singleton_insertion_cuts(Z1, (X0, Z0)) == (0,)


def test_tail_append_is_the_final_cut_specialization() -> None:
    blocks = (X1, Z0)
    cuts = singleton_insertion_cuts(X0, blocks)
    # X0 crosses X1 (disjoint -> commutes) but not Z0, so the final cut
    # (= tail append, position len(blocks)) is retained.
    assert cuts == (0, 2)
    assert len(blocks) in cuts


def test_empty_block_sequence_keeps_only_the_tail_cut() -> None:
    assert singleton_insertion_cuts(X0, ()) == (0,)


# ---------------------------------------------------------------------------
# Position-reduction parity with Paper I
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate, blocks",
    [
        (Z0, (X0, X1, Z0)),
        (X0, (Z0, Z0, Z0)),
        (Z1, (X0, Z0)),
        (MACRO_A, (MACRO_B, MACRO_MIXED, X0)),
        (MACRO_MIXED, (MACRO_A, MACRO_B)),
        (X0, ()),
    ],
)
def test_full_position_reduction_matches_paper_i_plan(candidate, blocks) -> None:
    positions = tuple(range(len(blocks) + 1))
    ours = reduce_insertion_positions(candidate, blocks, positions)
    paper_i = _commutation_reduced_insertion_position_plan(
        candidate_term=candidate,
        selected_ops=blocks,
        positions=positions,
    )
    assert ours == paper_i


def test_partial_position_request_matches_paper_i_plan() -> None:
    blocks = (X0, X1, Z0, MACRO_B)
    positions = (0, 2, 4)
    ours = reduce_insertion_positions(Z0, blocks, positions)
    paper_i = _commutation_reduced_insertion_position_plan(
        candidate_term=Z0,
        selected_ops=blocks,
        positions=positions,
    )
    assert ours == paper_i


def test_singleton_cuts_equal_full_range_representatives() -> None:
    for candidate, blocks in (
        (Z0, (X0, X1, Z0)),
        (MACRO_A, (MACRO_B, MACRO_MIXED, X0)),
        (X0, (Z0, X1)),
    ):
        plan = reduce_insertion_positions(
            candidate, blocks, tuple(range(len(blocks) + 1))
        )
        assert tuple(plan["representative_positions"]) == (
            singleton_insertion_cuts(candidate, blocks)
        )


def test_out_of_range_positions_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"must lie in \[0, 2\]"):
        reduce_insertion_positions(X0, (Z0, Z1), (0, 3))
