"""Generator-block commutation certificates and insertion cuts for AP-McLachlan.

The deletion-conditioned exchange selector inserts candidate children at
explicit circuit cuts, and collapses cuts that are related by certified
commuting swaps.  The commutation rule must be *identical* to Paper I's
insertion-position reduction, so this module owns no algebra: the word-level
parity test and the exact polynomial expansion both come from
:mod:`pipelines.static_adapt.commutation_metadata`, the single algebra source
both lanes share.  A cross-lane parity test
(``test_ap_mclachlan_commutation.py``) pins this module's cut reduction against
Paper I's ``_commutation_reduced_insertion_position_plan`` on shared fixtures,
so the semantics cannot drift apart silently.

Math:
    For a candidate generator ``a`` and ordered survivor blocks
    ``B_1, ..., B_n`` there are ``n + 1`` insertion cuts ``p = 0..n`` (cut
    ``p`` inserts after block ``p``; cut ``n`` is ordinary tail append).  Cuts
    ``p < q`` are equivalent when ``a`` commutes with every block crossed
    between them, so the retained earliest representatives are

        P(a) = {0} \\cup {j : 1 <= j <= n, [a, B_j] != 0}.

    Commutation of every nonzero Pauli component across two blocks is the
    sufficient exact certificate that swapping the implemented ordered
    rotation products leaves the unitary unchanged (the Paper-I rule).  A
    block whose expansion is empty or unavailable is conservatively treated as
    non-commuting: an uncertified swap is never performed.
"""

from __future__ import annotations

from typing import Any, Sequence

from pipelines.static_adapt.commutation_metadata import (
    AlgebraicMetadataError,
    expansion_from_ansatz_term,
    pauli_words_commute,
)


DEFAULT_COEFFICIENT_TOLERANCE = 1.0e-12


def generator_blocks_commute(
    left_term: Any,
    right_term: Any,
    *,
    coefficient_tolerance: float = DEFAULT_COEFFICIENT_TOLERANCE,
) -> bool:
    """Certify that two implemented generator blocks can be interchanged.

    Semantics match Paper I's ``_termwise_generators_commute_for_insertion``
    exactly: every retained Pauli component of one block must commute with
    every retained component of the other, retention is by coefficient
    magnitude above ``coefficient_tolerance``, and any failure to expand a
    block — or an empty retained expansion — returns ``False`` so the swap is
    never certified on incomplete evidence.
    """

    try:
        left = expansion_from_ansatz_term(left_term, key="left")
        right = expansion_from_ansatz_term(right_term, key="right")
    except AlgebraicMetadataError:
        return False
    left_terms = [
        term
        for term in left.terms
        if abs(complex(term.coeff)) > float(coefficient_tolerance)
    ]
    right_terms = [
        term
        for term in right.terms
        if abs(complex(term.coeff)) > float(coefficient_tolerance)
    ]
    if not left_terms or not right_terms:
        return False
    try:
        return all(
            pauli_words_commute(lhs.pauli_exyz, rhs.pauli_exyz)
            for lhs in left_terms
            for rhs in right_terms
        )
    except AlgebraicMetadataError:
        return False


def block_commutation_crossings(
    candidate_term: Any,
    blocks: Sequence[Any],
    *,
    coefficient_tolerance: float = DEFAULT_COEFFICIENT_TOLERANCE,
) -> tuple[bool, ...]:
    """Return, per ordered block, whether the candidate certifiably crosses it."""

    return tuple(
        generator_blocks_commute(
            candidate_term,
            block,
            coefficient_tolerance=coefficient_tolerance,
        )
        for block in blocks
    )


def singleton_insertion_cuts(
    candidate_term: Any,
    blocks: Sequence[Any],
    *,
    coefficient_tolerance: float = DEFAULT_COEFFICIENT_TOLERANCE,
) -> tuple[int, ...]:
    """Return the retained earliest-representative cuts ``P(a)``.

    ``blocks`` is the ordered survivor sequence ``B_1..B_n``; the result is a
    sorted tuple containing ``0`` plus every ``j`` whose block the candidate
    does not certifiably commute with.  Cut ``len(blocks)`` appears exactly
    when the final block is uncrossable, which is the tail-append case.
    """

    crossings = block_commutation_crossings(
        candidate_term,
        blocks,
        coefficient_tolerance=coefficient_tolerance,
    )
    cuts = [0]
    for index, commutes in enumerate(crossings):
        if not commutes:
            cuts.append(index + 1)
    return tuple(cuts)


def reduce_insertion_positions(
    candidate_term: Any,
    blocks: Sequence[Any],
    positions: Sequence[int],
    *,
    coefficient_tolerance: float = DEFAULT_COEFFICIENT_TOLERANCE,
) -> dict[str, Any]:
    """Collapse requested cuts into earliest commuting-class representatives.

    Mirrors the Paper-I plan payload (schema
    ``commutation_reduced_insertion_positions_v1``): each position belongs to
    the class starting at the first cut reachable from it by certified
    commuting swaps, and each class keeps its smallest requested member.
    """

    n_blocks = int(len(blocks))
    requested = sorted({int(position) for position in positions})
    invalid = [p for p in requested if p < 0 or p > n_blocks]
    if invalid:
        raise ValueError(
            f"Insertion positions must lie in [0, {n_blocks}], got {invalid}."
        )

    crossings = block_commutation_crossings(
        candidate_term,
        blocks,
        coefficient_tolerance=coefficient_tolerance,
    )
    class_start_by_position: dict[int, int] = {0: 0}
    class_start = 0
    for index, commutes in enumerate(crossings):
        if not commutes:
            class_start = int(index + 1)
        class_start_by_position[int(index + 1)] = int(class_start)

    requested_by_class: dict[int, list[int]] = {}
    for position in requested:
        requested_by_class.setdefault(
            int(class_start_by_position[int(position)]), []
        ).append(int(position))
    members_by_representative: dict[int, list[int]] = {}
    representative_by_position: dict[int, int] = {}
    for members in requested_by_class.values():
        representative = int(min(members))
        members_by_representative[representative] = [int(x) for x in members]
        for position in members:
            representative_by_position[int(position)] = int(representative)
    representatives = sorted(int(x) for x in members_by_representative)
    return {
        "schema": "commutation_reduced_insertion_positions_v1",
        "requested_positions": [int(x) for x in requested],
        "representative_positions": [int(x) for x in representatives],
        "representative_by_position": {
            int(key): int(value)
            for key, value in representative_by_position.items()
        },
        "members_by_representative": {
            int(key): [int(x) for x in value]
            for key, value in members_by_representative.items()
        },
        "commuting_crossings": [bool(x) for x in crossings],
        "collapsed_position_count": int(len(requested) - len(representatives)),
    }


__all__ = [
    "DEFAULT_COEFFICIENT_TOLERANCE",
    "block_commutation_crossings",
    "generator_blocks_commute",
    "reduce_insertion_positions",
    "singleton_insertion_cuts",
]
