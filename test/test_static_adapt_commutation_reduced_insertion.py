from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np
import pytest

from pipelines.scaffold import hh_continuation_scoring as scoring
from pipelines.static_adapt.adapt_pipeline import (
    _candidate_insertion_position_plans,
    _commutation_reduced_insertion_position_plan,
    _phase1_position_probe_plan,
)
from pipelines.static_adapt.ra_adapt.insertion_geometry import (
    APPEND_COMMUTATION_REDUCED_MODE,
    APPEND_COMMUTATION_REDUCED_POLICY,
    APPEND_ENDPOINT_POSITION_SCOPE,
    EXACT_TERM_COMMUTATION_EQUIVALENCE,
    append_commutation_reduced_position_domain,
    append_commutation_reduced_receipt_header,
    enumerate_candidate_position_plans,
    validate_commutation_reduced_insertion_receipt,
)
from pipelines.static_adapt.sr_snake import (
    AppendCommutationReducedInsertion,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AppendOnlyInsertion,
    PlateauCommutationInsertion,
    SRMethodPolicy,
)
from pipelines.scaffold.hh_continuation_stage_control import StageControllerConfig
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _term(label: str, *words: str) -> AnsatzTerm:
    nq = len(words[0])
    return AnsatzTerm(
        label=label,
        polynomial=PauliPolynomial(
            "JW",
            [PauliTerm(nq, ps=word, pc=1.0) for word in words],
        ),
    )


def test_commutation_reduction_keeps_one_representative_per_contiguous_class() -> None:
    candidate = _term("candidate", "ex")
    selected = [
        _term("commuting-left", "ex"),
        _term("barrier", "ez"),
        _term("commuting-right", "xx"),
    ]

    plan = _commutation_reduced_insertion_position_plan(
        candidate_term=candidate,
        selected_ops=selected,
        positions=[0, 1, 2, 3],
    )

    assert plan["representative_positions"] == [0, 2]
    assert plan["members_by_representative"] == {0: [0, 1], 2: [2, 3]}
    assert plan["commuting_crossings"] == [True, False, True]
    assert plan["collapsed_position_count"] == 2


def test_existing_generators_need_not_commute_with_each_other() -> None:
    candidate = _term("candidate", "ez")
    selected = [
        _term("left", "xz"),
        _term("right", "yz"),
    ]

    plan = _commutation_reduced_insertion_position_plan(
        candidate_term=candidate,
        selected_ops=selected,
        positions=[0, 1, 2],
    )

    assert plan["representative_positions"] == [0]
    assert plan["members_by_representative"] == {0: [0, 1, 2]}


def test_macro_commutation_certificate_requires_all_cross_components_to_commute() -> None:
    candidate = _term("candidate", "ex", "ey")
    selected = [_term("existing", "ex")]

    plan = _commutation_reduced_insertion_position_plan(
        candidate_term=candidate,
        selected_ops=selected,
        positions=[0, 1],
    )

    assert plan["representative_positions"] == [0, 1]
    assert plan["commuting_crossings"] == [False]


def test_commutation_reduced_mode_starts_from_the_full_position_domain() -> None:
    positions, triggered, reason = _phase1_position_probe_plan(
        insertion_mode="full_commutation_reduced",
        append_eval={},
        append_position=15,
        n_params=15,
        active_window_indices=[12, 13, 14],
        stage_name="core",
        drop_plateau_hits=0,
        max_grad=1.0,
        eps_grad=1.0e-8,
        finite_angle_fallback=False,
        repeated_family_flat=False,
        cfg=StageControllerConfig(max_probe_positions=2),
    )

    assert positions == list(range(16))
    assert triggered is True
    assert reason == "full_commutation_reduced"


def test_candidate_position_planner_has_no_unreduced_bypass() -> None:
    assert "commutation_reduce" not in inspect.signature(
        _candidate_insertion_position_plans
    ).parameters


def test_append_reduced_policy_is_typed_without_changing_historical_append() -> None:
    policy = AppendCommutationReducedInsertion()

    assert policy.to_dict() == {"kind": "append_commutation_reduced"}
    assert SRMethodPolicy(insertion=policy).insertion is policy
    assert policy.runtime_mode == APPEND_COMMUTATION_REDUCED_MODE
    assert policy.position_scope == APPEND_ENDPOINT_POSITION_SCOPE
    assert policy.equivalence_policy == EXACT_TERM_COMMUTATION_EQUIVALENCE
    assert policy.receipt_key == "insertion_commutation_reduced"
    assert AppendOnlyInsertion().to_dict() == {"kind": "append_only"}
    assert isinstance(SRMethodPolicy().insertion, PlateauCommutationInsertion)
    assert APPEND_ENDPOINT_POSITION_SCOPE == (
        "append_endpoint_only_every_depth_v1"
    )
    assert EXACT_TERM_COMMUTATION_EQUIVALENCE == (
        "termwise_cross_component_commutation_earliest_representative_v1"
    )


def test_append_reduced_domain_executes_exact_reducer_at_endpoint() -> None:
    candidate = _term("candidate", "ex")
    selected = [
        _term("commuting-left", "ex"),
        _term("commuting-middle", "xx"),
        _term("commuting-right", "ex"),
    ]
    domain = append_commutation_reduced_position_domain(
        append_position=len(selected),
        n_params=len(selected),
    )

    plans = enumerate_candidate_position_plans(
        pool=[candidate],
        candidate_indices=[0],
        selected_ops=selected,
        domain=domain,
    )

    assert domain.positions == (len(selected),)
    assert domain.reason == APPEND_COMMUTATION_REDUCED_MODE
    assert plans[0]["requested_positions"] == [len(selected)]
    assert plans[0]["representative_positions"] == [len(selected)]
    assert plans[0]["representative_by_position"] == {
        len(selected): len(selected)
    }
    assert plans[0]["members_by_representative"] == {
        len(selected): [len(selected)]
    }
    assert plans[0]["commuting_crossings"] == [True, True, True]
    assert plans[0]["collapsed_position_count"] == 0


def test_retired_raw_full_mode_fails_closed() -> None:
    with pytest.raises(
        ValueError,
        match="raw full insertion mode is retired",
    ):
        _phase1_position_probe_plan(
            insertion_mode="full",
            append_eval={},
            append_position=3,
            n_params=3,
            active_window_indices=[2],
            stage_name="core",
            drop_plateau_hits=0,
            max_grad=1.0,
            eps_grad=1.0e-8,
            finite_angle_fallback=False,
            repeated_family_flat=False,
            cfg=StageControllerConfig(max_probe_positions=4),
        )


def _reduced_receipt_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    plans = [
        {
            "schema": "commutation_reduced_insertion_positions_v1",
            "candidate_pool_index": 0,
            "candidate_label": "commuting",
            "requested_positions": [0, 1],
            "representative_positions": [0],
            "representative_by_position": {0: 0, 1: 0},
            "members_by_representative": {0: [0, 1]},
            "commuting_crossings": [True],
            "collapsed_position_count": 1,
        },
        {
            "schema": "commutation_reduced_insertion_positions_v1",
            "candidate_pool_index": 2,
            "candidate_label": "noncommuting",
            "requested_positions": [0, 1],
            "representative_positions": [0, 1],
            "representative_by_position": {0: 0, 1: 1},
            "members_by_representative": {0: [0], 1: [1]},
            "commuting_crossings": [False],
            "collapsed_position_count": 0,
        },
    ]
    receipt = {
        "schema": "commutation_reduced_insertion_domain_receipt_v1",
        "policy": "always_commutation_reduced",
        "domain_state": "open",
        "domain_open": True,
        "effective_insertion_mode": "full_commutation_reduced",
        "requested_positions": [0, 1],
        "requested_position_count": 2,
        "candidate_count": 2,
        "retained_representative_count": 3,
        "collapsed_position_count": 1,
        "candidate_position_plans": plans,
        "retained_representatives": [
            {
                "candidate_pool_index": 0,
                "candidate_label": "commuting",
                "positions": [0],
            },
            {
                "candidate_pool_index": 2,
                "candidate_label": "noncommuting",
                "positions": [0, 1],
            },
        ],
    }
    scored = {
        "append_position": 1,
        "phases": [
            {
                "phase": "phase_i",
                "records": [
                    {"pool_index": 0, "insertion_position": 0},
                    {"pool_index": 2, "insertion_position": 0},
                    {"pool_index": 2, "insertion_position": 1},
                ],
            }
        ],
    }
    return receipt, scored


def _append_reduced_receipt_fixture() -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    append_position = 3
    receipt = {
        **append_commutation_reduced_receipt_header(
            append_position=append_position,
        ),
        "requested_positions": [append_position],
        "requested_position_count": 1,
        "candidate_count": 1,
        "retained_representative_count": 1,
        "collapsed_position_count": 0,
        "candidate_position_plans": [
            {
                "schema": "commutation_reduced_insertion_positions_v1",
                "candidate_pool_index": 4,
                "candidate_label": "endpoint",
                "requested_positions": [append_position],
                "representative_positions": [append_position],
                "representative_by_position": {
                    append_position: append_position
                },
                "members_by_representative": {
                    append_position: [append_position]
                },
                "commuting_crossings": [True, False, True],
                "collapsed_position_count": 0,
            }
        ],
        "retained_representatives": [
            {
                "candidate_pool_index": 4,
                "candidate_label": "endpoint",
                "positions": [append_position],
            }
        ],
    }
    scored = {
        "append_position": append_position,
        "phases": [
            {
                "phase": "phase_i",
                "records": [
                    {
                        "pool_index": 4,
                        "insertion_position": append_position,
                    }
                ],
            }
        ],
    }
    return receipt, scored


def test_append_reduced_receipt_authenticates_endpoint_reduction() -> None:
    receipt, scored = _append_reduced_receipt_fixture()

    validated = validate_commutation_reduced_insertion_receipt(
        receipt,
        expected_policy=APPEND_COMMUTATION_REDUCED_POLICY,
        expected_requested_positions=[3],
        scored_population=scored,
    )

    assert validated["schema"] == (
        "commutation_reduced_insertion_domain_receipt_v1"
    )
    assert validated["domain_open"] is False
    assert validated["retained_representative_count"] == 1
    assert validated["collapsed_position_count"] == 0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda receipt: receipt.update({"append_position": 2}),
            "append endpoint",
        ),
        (
            lambda receipt: receipt.update({"domain_open": True}),
            "domain state",
        ),
        (
            lambda receipt: receipt.pop("append_position"),
            "append_position",
        ),
    ],
    ids=[
        "non-endpoint-request",
        "opened-domain",
        "missing-endpoint",
    ],
)
def test_append_reduced_receipt_rejects_semantic_drift(
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    receipt, scored = _append_reduced_receipt_fixture()
    mutation(receipt)

    with pytest.raises((TypeError, ValueError), match=match):
        validate_commutation_reduced_insertion_receipt(
            receipt,
            expected_policy=APPEND_COMMUTATION_REDUCED_POLICY,
            scored_population=scored,
        )


def test_reduced_receipt_validator_closes_requested_and_scored_domains() -> None:
    receipt, scored = _reduced_receipt_fixture()

    validated = validate_commutation_reduced_insertion_receipt(
        receipt,
        expected_policy="always_commutation_reduced",
        expected_requested_positions=[0, 1],
        scored_population=scored,
    )

    assert validated["retained_representative_count"] == 3
    assert validated["collapsed_position_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda receipt, _scored: receipt[
                "candidate_position_plans"
            ][0].update(
                {
                    "representative_positions": [1],
                    "representative_by_position": {0: 1, 1: 1},
                    "members_by_representative": {1: [0, 1]},
                }
            ),
            "earliest",
        ),
        (
            lambda receipt, _scored: receipt[
                "candidate_position_plans"
            ][0].update(
                {"schema": "unreduced_insertion_positions_v1"}
            ),
            "schema",
        ),
        (
            lambda receipt, _scored: receipt.update(
                {"retained_representative_count": 4}
            ),
            "count",
        ),
        (
            lambda receipt, _scored: receipt[
                "candidate_position_plans"
            ][0].update({"commuting_crossings": [False]}),
            "crossing",
        ),
        (
            lambda _receipt, scored: scored["phases"][0]["records"].append(
                {"pool_index": 0, "insertion_position": 1}
            ),
            "scored",
        ),
    ],
    ids=[
        "non-earliest-representative",
        "unreduced-plan",
        "aggregate-count-drift",
        "crossing-class-mismatch",
        "raw-position-scored",
    ],
)
def test_reduced_receipt_validator_rejects_tampering(
    mutation: Callable[[dict[str, Any], dict[str, Any]], None],
    match: str,
) -> None:
    receipt, scored = _reduced_receipt_fixture()
    mutation(receipt, scored)

    with pytest.raises(ValueError, match=match):
        validate_commutation_reduced_insertion_receipt(
            receipt,
            expected_policy="always_commutation_reduced",
            expected_requested_positions=[0, 1],
            scored_population=scored,
        )


def test_exact_first_order_context_resolves_noncommuting_insertion_positions() -> None:
    rng = np.random.default_rng(20260725)
    psi_ref = np.asarray(
        rng.normal(size=4) + 1.0j * rng.normal(size=4),
        dtype=complex,
    )
    psi_ref /= np.linalg.norm(psi_ref)
    selected = [_term("selected-x", "xe"), _term("selected-yz", "yz")]
    candidate = _term("candidate-zx", "zx")
    theta = np.asarray([0.17, -0.23], dtype=float)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    h_compiled = compile_polynomial_action(
        PauliPolynomial(
            "JW",
            [
                PauliTerm(2, ps="ze", pc=0.7),
                PauliTerm(2, ps="ex", pc=-0.4),
                PauliTerm(2, ps="xx", pc=0.3),
            ],
        ),
        pauli_action_cache={},
    )
    hpsi_state = apply_compiled_polynomial(psi_state, h_compiled)
    candidate_compiled = compile_polynomial_action(
        candidate.polynomial,
        pauli_action_cache={},
    )
    context = scoring._prepare_exact_insertion_first_order_context(
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        hpsi_state=hpsi_state,
        pauli_action_cache={},
        state_consistency_tolerance=1e-10,
    )

    inserted = scoring._exact_insertion_first_order_candidate_geometry(
        context=context,
        candidate_term=candidate,
        position_id=0,
        candidate_compiled=candidate_compiled,
    )
    appended = scoring._exact_insertion_first_order_candidate_geometry(
        context=context,
        candidate_term=candidate,
        position_id=len(selected),
        candidate_compiled=candidate_compiled,
    )
    append_action = apply_compiled_polynomial(psi_state, candidate_compiled)
    append_gradient = adapt_commutator_grad_from_hpsi(
        hpsi_state,
        append_action,
    )
    append_mean = complex(np.vdot(psi_state, append_action))
    append_centered = append_action - append_mean * psi_state
    append_metric = float(np.real(np.vdot(append_centered, append_centered)))

    assert appended["energy_gradient"] == pytest.approx(
        append_gradient,
        abs=1e-12,
    )
    assert appended["fubini_study_metric"] == pytest.approx(
        append_metric,
        abs=1e-12,
    )
    assert inserted["energy_gradient"] != pytest.approx(
        appended["energy_gradient"],
        abs=1e-8,
    )
