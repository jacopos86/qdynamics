"""Pure mathematical contract for Paper-II generalized exchange."""

from pipelines.time_dynamics.generalized_exchange import (
    EXCHANGE_FACE_DELETE_ONLY,
    EXCHANGE_FACE_FULL,
    EXCHANGE_FACE_INSERT_ONLY,
    EXCHANGE_RANKING_COST_AWARE,
    EXCHANGE_RANKING_SIGNED_DRIFT,
    GeneralizedExchange,
    GeneralizedPatch,
    REALIZED_ACCEPT,
    REALIZED_REFUSE,
    REALIZED_RETRY_INSERT_FACE,
)


def test_insert_delete_and_true_exchange_are_faces_of_one_patch() -> None:
    assert GeneralizedPatch(insertions=(("A", 2),)).kind == "insert"
    assert GeneralizedPatch(deletions=(3,)).kind == "delete"
    assert GeneralizedPatch(deletions=(3,), insertions=(("A", 2),)).kind == "exchange"
    assert GeneralizedPatch().kind == "stay"


def test_drift_ranked_debt_opens_the_complete_family() -> None:
    rule = GeneralizedExchange(
        l2_cut=1.0e-3,
        debt_policy="drift_ranked",
        support_floor=1,
        insertion_cardinality_cap=1,
    )
    domain = rule.domain(
        checkpoint_l2=2.0e-3,
        insertion_gate_open=True,
        deletion_candidate_count=4,
    )
    assert domain.face == EXCHANGE_FACE_FULL
    assert domain.true_exchange_face_open is True
    assert domain.ranking == EXCHANGE_RANKING_SIGNED_DRIFT


def test_below_the_cut_is_the_measurement_free_delete_face() -> None:
    rule = GeneralizedExchange(l2_cut=1.0e-3)
    domain = rule.domain(
        checkpoint_l2=5.0e-4,
        insertion_gate_open=False,
        deletion_candidate_count=3,
    )
    assert domain.face == EXCHANGE_FACE_DELETE_ONLY
    assert domain.ranking == EXCHANGE_RANKING_COST_AWARE


def test_insert_only_is_an_explicit_debt_ablation() -> None:
    rule = GeneralizedExchange(l2_cut=1.0e-3, debt_policy="insertion_only")
    domain = rule.domain(
        checkpoint_l2=2.0e-3,
        insertion_gate_open=True,
        deletion_candidate_count=4,
    )
    assert domain.face == EXCHANGE_FACE_INSERT_ONLY
    assert domain.deletion_face_open is False


def test_caps_close_faces_without_creating_another_algorithm() -> None:
    rule = GeneralizedExchange(l2_cut=1.0e-3, insertion_cardinality_cap=0)
    domain = rule.domain(
        checkpoint_l2=2.0e-3,
        insertion_gate_open=True,
        deletion_candidate_count=2,
    )
    assert domain.face == EXCHANGE_FACE_DELETE_ONLY


def test_realized_l2_rule_accepts_retries_or_refuses() -> None:
    rule = GeneralizedExchange(l2_cut=1.0e-3)
    full = rule.domain(
        checkpoint_l2=2.0e-3,
        insertion_gate_open=True,
        deletion_candidate_count=2,
    )
    exchange = GeneralizedPatch(deletions=(0,), insertions=(("A", 1),))
    insert = GeneralizedPatch(insertions=(("A", 1),))

    assert rule.assess_realized_candidate(
        domain=full, patch=exchange, candidate_l2=1.0e-3
    ) == REALIZED_ACCEPT
    assert rule.assess_realized_candidate(
        domain=full, patch=exchange, candidate_l2=3.0e-3
    ) == REALIZED_RETRY_INSERT_FACE
    assert rule.assess_realized_candidate(
        domain=full, patch=insert, candidate_l2=3.0e-3
    ) == REALIZED_REFUSE
