from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.sr_snake import (
    AcceptedStateResume,
    AlwaysCommutationReducedInsertion,
    AppendCommutationReducedInsertion,
    CheckpointObservation,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRObservationPolicy,
    SRRunRequest,
    SRStopPolicy,
    run_sr_snake,
)
from pipelines.static_adapt.sr_snake._resume import (
    CanonicalResumeError,
    _resolve_authenticated_insertion_policy,
    _validate_append_only_scored_population,
    _validate_commutation_reduced_insertion_round,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _small_hh_problem():
    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        ),
        exact_energy_impl=adapt_pipeline._exact_gs_energy_for_problem,
    )


def _valid_reduced_receipt() -> dict[str, Any]:
    plans = [
        {
            "candidate_pool_index": 0,
            "candidate_label": "candidate:commuting",
            "schema": "commutation_reduced_insertion_positions_v1",
            "requested_positions": [0, 1],
            "representative_positions": [0],
            "representative_by_position": {"0": 0, "1": 0},
            "members_by_representative": {"0": [0, 1]},
            "commuting_crossings": [True],
            "collapsed_position_count": 1,
        },
        {
            "candidate_pool_index": 2,
            "candidate_label": "candidate:barrier",
            "schema": "commutation_reduced_insertion_positions_v1",
            "requested_positions": [0, 1],
            "representative_positions": [0, 1],
            "representative_by_position": {"0": 0, "1": 1},
            "members_by_representative": {
                "0": [0],
                "1": [1],
            },
            "commuting_crossings": [False],
            "collapsed_position_count": 0,
        },
    ]
    return {
        "schema": "commutation_reduced_insertion_domain_receipt_v1",
        "policy": "always_commutation_reduced",
        "domain_state": "open",
        "domain_open": True,
        "effective_insertion_mode": "full_commutation_reduced",
        "requested_positions": [0, 1],
        "retained_representatives": [
            {
                "candidate_pool_index": 0,
                "candidate_label": "candidate:commuting",
                "positions": [0],
            },
            {
                "candidate_pool_index": 2,
                "candidate_label": "candidate:barrier",
                "positions": [0, 1],
            },
        ],
        "candidate_position_plans": plans,
        "candidate_count": 2,
        "requested_position_count": 2,
        "retained_representative_count": 3,
        "collapsed_position_count": 1,
    }


def _scored_population() -> dict[str, Any]:
    phase_rows = {
        "phase_i": [
            (0, 0, "generator:0"),
            (2, 0, "generator:2"),
            (2, 1, "generator:2"),
        ],
        "phase_ii": [(2, 1, "generator:2")],
        "phase_iii": [(2, 1, "generator:2")],
    }
    phases = []
    all_records: list[dict[str, Any]] = []
    for phase_name, rows in phase_rows.items():
        records = [
            {
                "domain_record_id": (
                    f"{phase_name}:pool={pool_index}:position={position}"
                ),
                "generator_id": generator_id,
                "pool_index": pool_index,
                "pool_label": f"candidate:{pool_index}",
                "insertion_position": position,
                "position_class": "interior" if position < 1 else "append",
            }
            for pool_index, position, generator_id in rows
        ]
        phases.append(
            {
                "phase": phase_name,
                "population_count": len(records),
                "records": records,
                "ordered_population_sha256": _digest(records),
            }
        )
        all_records.extend(records)
    payload = {
        "schema": "paper_i_scored_insertion_position_population_v1",
        "coordinate_chart": "exact_ordered_insertion_zero_angle_v1",
        "append_position": 1,
        "phase_order": ["phase_i", "phase_ii", "phase_iii"],
        "phases": phases,
        "scored_record_count": len(all_records),
        "interior_scored_count": sum(
            record["position_class"] == "interior"
            for record in all_records
        ),
        "append_scored_count": sum(
            record["position_class"] == "append"
            for record in all_records
        ),
    }
    payload["sha256"] = _digest(payload)
    return payload


def _phase0_scored_population() -> dict[str, Any]:
    def _record(
        pool_index: int,
        position: int,
        generator_id: str,
        pool_label: str,
    ) -> dict[str, Any]:
        return {
            "domain_record_id": (
                f"candidate:pool={pool_index}:position={position}"
            ),
            "generator_id": generator_id,
            "pool_index": pool_index,
            "pool_label": pool_label,
            "insertion_position": position,
            "position_class": "interior" if position < 1 else "append",
        }

    population = [
        _record(
            0,
            0,
            "generator:0::pool[0]",
            "candidate:commuting",
        ),
        _record(
            2,
            0,
            "generator:2::pool[2]",
            "candidate:barrier",
        ),
        _record(
            2,
            1,
            "generator:2::pool[2]",
            "candidate:barrier",
        ),
    ]
    shortlist = copy.deepcopy(population[1:])
    phase_i_shortlist = copy.deepcopy(shortlist)
    for record in phase_i_shortlist:
        record["generator_id"] = str(record["generator_id"]).split(
            "::pool[", maxsplit=1
        )[0]
    phase_rows = (
        ("phase_i", phase_i_shortlist),
        ("phase_ii", phase_i_shortlist[1:]),
        ("phase_iii", phase_i_shortlist[1:]),
    )
    phases: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for phase_name, rows in phase_rows:
        records = copy.deepcopy(rows)
        phases.append(
            {
                "phase": phase_name,
                "population_count": len(records),
                "records": records,
                "ordered_population_sha256": _digest(records),
            }
        )
        all_records.extend(records)
    payload = {
        "schema": "paper_i_scored_insertion_position_population_v1",
        "coordinate_chart": "exact_ordered_insertion_zero_angle_v1",
        "append_position": 1,
        "phase_order": ["phase_i", "phase_ii", "phase_iii"],
        "phases": phases,
        "scored_record_count": len(all_records),
        "interior_scored_count": sum(
            record["position_class"] == "interior"
            for record in all_records
        ),
        "append_scored_count": sum(
            record["position_class"] == "append"
            for record in all_records
        ),
        "phase0_gradient_screen": {
            "schema": "paper_i_scored_gradient_phase0_population_v1",
            "population_count": len(population),
            "population": population,
            "ordered_population_sha256": _digest(population),
            "shortlist_count": len(shortlist),
            "shortlist": shortlist,
            "ordered_shortlist_sha256": _digest(shortlist),
        },
    }
    payload["sha256"] = _digest(payload)
    return payload


def _append_reduced_receipt() -> dict[str, Any]:
    return {
        "schema": "commutation_reduced_insertion_domain_receipt_v1",
        "policy": "append_commutation_reduced",
        "domain_state": "closed",
        "domain_open": False,
        "effective_insertion_mode": "append_commutation_reduced",
        "append_position": 1,
        "requested_positions": [1],
        "retained_representatives": [
            {
                "candidate_pool_index": 2,
                "candidate_label": "candidate:2",
                "positions": [1],
            }
        ],
        "candidate_position_plans": [
            {
                "candidate_pool_index": 2,
                "candidate_label": "candidate:2",
                "schema": "commutation_reduced_insertion_positions_v1",
                "requested_positions": [1],
                "representative_positions": [1],
                "representative_by_position": {"1": 1},
                "members_by_representative": {"1": [1]},
                "commuting_crossings": [True],
                "collapsed_position_count": 0,
            }
        ],
        "candidate_count": 1,
        "requested_position_count": 1,
        "retained_representative_count": 1,
        "collapsed_position_count": 0,
    }


def _append_reduced_scored_population() -> dict[str, Any]:
    phases: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for phase_name in ("phase_i", "phase_ii", "phase_iii"):
        records = [
            {
                "domain_record_id": (
                    f"{phase_name}:pool=2:position=1"
                ),
                "generator_id": "generator:2",
                "pool_index": 2,
                "pool_label": "candidate:2",
                "insertion_position": 1,
                "position_class": "append",
            }
        ]
        phases.append(
            {
                "phase": phase_name,
                "population_count": 1,
                "records": records,
                "ordered_population_sha256": _digest(records),
            }
        )
        all_records.extend(records)
    payload = {
        "schema": "paper_i_scored_insertion_position_population_v1",
        "coordinate_chart": "exact_ordered_insertion_zero_angle_v1",
        "append_position": 1,
        "phase_order": ["phase_i", "phase_ii", "phase_iii"],
        "phases": phases,
        "scored_record_count": len(all_records),
        "interior_scored_count": 0,
        "append_scored_count": len(all_records),
    }
    payload["sha256"] = _digest(payload)
    return payload


def _validate(
    receipt: dict[str, Any],
    scored: dict[str, Any] | None = None,
) -> None:
    _validate_commutation_reduced_insertion_round(
        receipt,
        owner="history[1] insertion",
        expected_schema=(
            "commutation_reduced_insertion_domain_receipt_v1"
        ),
        expected_policy="always_commutation_reduced",
        expected_requested_positions=(0, 1),
        expected_domain_open=True,
        scored_population=(
            _scored_population() if scored is None else scored
        ),
    )


def test_resume_accepts_exact_commutation_reduction_closure() -> None:
    plans = _validate_commutation_reduced_insertion_round(
        _valid_reduced_receipt(),
        owner="history[1] insertion",
        expected_schema=(
            "commutation_reduced_insertion_domain_receipt_v1"
        ),
        expected_policy="always_commutation_reduced",
        expected_requested_positions=(0, 1),
        expected_domain_open=True,
        scored_population=_scored_population(),
    )

    assert tuple(plans) == (0, 2)
    assert tuple(plans[0]["representative_positions"]) == (0,)
    assert tuple(plans[2]["representative_positions"]) == (0, 1)


def test_resume_accepts_authenticated_phase0_shortlist_as_phase_i() -> None:
    plans = _validate_commutation_reduced_insertion_round(
        _valid_reduced_receipt(),
        owner="history[1] insertion",
        expected_schema=(
            "commutation_reduced_insertion_domain_receipt_v1"
        ),
        expected_policy="always_commutation_reduced",
        expected_requested_positions=(0, 1),
        expected_domain_open=True,
        scored_population=_phase0_scored_population(),
    )

    assert tuple(plans) == (0, 2)


def test_resume_authenticates_append_endpoint_reduction_closure() -> None:
    plans = _validate_commutation_reduced_insertion_round(
        _append_reduced_receipt(),
        owner="history[1] append-reduced insertion",
        expected_schema=(
            "commutation_reduced_insertion_domain_receipt_v1"
        ),
        expected_policy=AppendCommutationReducedInsertion.kind,
        expected_requested_positions=(1,),
        expected_domain_open=False,
        expected_effective_mode=(
            AppendCommutationReducedInsertion.runtime_mode
        ),
        scored_population=_append_reduced_scored_population(),
    )

    assert tuple(plans) == (2,)
    assert tuple(plans[2]["representative_positions"]) == (1,)
    assert plans[2]["collapsed_position_count"] == 0


def test_resume_append_reduced_policy_requires_scope_and_equivalence() -> None:
    route = {
        "semantic_invariants": {
            "canonical_insertion_policy": (
                AppendCommutationReducedInsertion.kind
            ),
            "insertion_position_scope": (
                AppendCommutationReducedInsertion.position_scope
            ),
            "insertion_equivalence_policy": (
                AppendCommutationReducedInsertion.equivalence_policy
            ),
        },
        "execution_settings": {
            "adapt_insertion_mode": (
                AppendCommutationReducedInsertion.runtime_mode
            )
        },
    }

    assert _resolve_authenticated_insertion_policy(route) == (
        AppendCommutationReducedInsertion.kind
    )
    route["semantic_invariants"]["insertion_position_scope"] = (
        "append_only_unreduced_v1"
    )
    with pytest.raises(CanonicalResumeError, match="endpoint"):
        _resolve_authenticated_insertion_policy(route)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt["candidate_position_plans"][0].__setitem__(
            "schema", "unreduced_insertion_positions_v1"
        ),
        lambda receipt: receipt["candidate_position_plans"][0].__setitem__(
            "requested_positions", [0]
        ),
        lambda receipt: receipt["candidate_position_plans"][0].__setitem__(
            "representative_positions", [1]
        ),
        lambda receipt: receipt["candidate_position_plans"][0].__setitem__(
            "members_by_representative", {"0": [0], "1": [1]}
        ),
        lambda receipt: receipt["candidate_position_plans"][0].__setitem__(
            "representative_by_position", {"0": 0, "1": 1}
        ),
        lambda receipt: receipt["candidate_position_plans"][0].__setitem__(
            "collapsed_position_count", 0
        ),
        lambda receipt: receipt["candidate_position_plans"][0].__setitem__(
            "commuting_crossings", [False]
        ),
        lambda receipt: receipt.__setitem__(
            "retained_representative_count", 4
        ),
        lambda receipt: receipt.__setitem__(
            "requested_positions", [0]
        ),
    ],
    ids=[
        "unreduced-schema",
        "candidate-domain-hole",
        "nonmember-representative",
        "nonpartitioning-members",
        "inconsistent-representative-map",
        "candidate-collapse-count",
        "crossing-class-mismatch",
        "global-representative-count",
        "global-domain-hole",
    ],
)
def test_resume_rejects_tampered_commutation_reduction_closure(
    mutate,
) -> None:
    receipt = copy.deepcopy(_valid_reduced_receipt())
    mutate(receipt)

    with pytest.raises(CanonicalResumeError):
        _validate(receipt)


def test_resume_rejects_non_earliest_class_representative() -> None:
    receipt = copy.deepcopy(_valid_reduced_receipt())
    plan = receipt["candidate_position_plans"][0]
    plan["representative_positions"] = [1]
    plan["representative_by_position"] = {"0": 1, "1": 1}
    plan["members_by_representative"] = {"1": [0, 1]}
    receipt["retained_representatives"][0]["positions"] = [1]

    with pytest.raises(CanonicalResumeError, match="earliest"):
        _validate(receipt)


def test_resume_rejects_scored_position_outside_representatives() -> None:
    scored = _scored_population()
    scored["phases"][0]["records"][0]["insertion_position"] = 1

    with pytest.raises(CanonicalResumeError, match="scored"):
        _validate(_valid_reduced_receipt(), scored)


def test_append_only_resume_rejects_any_interior_scored_position() -> None:
    with pytest.raises(CanonicalResumeError, match="append-only"):
        _validate_append_only_scored_population(
            _scored_population(),
            owner="history[1] scored population",
            append_position=1,
        )


@pytest.mark.parametrize(
    "semantic_invariants, execution_settings",
    [
        ({"canonical_insertion_policy": "full_commutation"}, {}),
        ({}, {"adapt_insertion_mode": "full"}),
        (
            {"canonical_insertion_policy": "always_commutation_reduced"},
            {"adapt_insertion_mode": "full"},
        ),
    ],
)
def test_resume_rejects_retired_unreduced_insertion_identity(
    semantic_invariants: dict[str, Any],
    execution_settings: dict[str, Any],
) -> None:
    with pytest.raises(CanonicalResumeError, match="retired"):
        _resolve_authenticated_insertion_policy(
            {
                "semantic_invariants": semantic_invariants,
                "execution_settings": execution_settings,
            }
        )


def test_live_always_reduced_checkpoint_resumes_and_rejects_missing_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )
    checkpoint_path = tmp_path / "always-reduced-current.json"
    problem = _small_hh_problem()
    method = SRMethodPolicy(
        insertion=AlwaysCommutationReducedInsertion()
    )
    first = run_sr_snake(
        problem,
        SRRunRequest(
            method=method,
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(path=checkpoint_path)
            ),
        ),
    )
    checkpoint_sha256 = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    resumed = run_sr_snake(
        problem,
        SRRunRequest(
            method=method,
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=2),
                resume=AcceptedStateResume(
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha256,
                ),
            ),
        ),
    )

    assert tuple(
        state.controller_round for state in resumed.accepted_trajectory
    ) == (1, 2)
    assert resumed.accepted_trajectory[0] == first.accepted_trajectory[0]
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["adapt_vqe"]["history"][0].pop(
        "insertion_commutation_reduced"
    )
    checkpoint["adapt_vqe"]["history_tail"] = copy.deepcopy(
        checkpoint["adapt_vqe"]["history"]
    )
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered_sha256 = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    with pytest.raises(
        CanonicalResumeError,
        match="insertion_commutation_reduced",
    ):
        run_sr_snake(
            problem,
            SRRunRequest(
                method=method,
                execution=SRExecutionPolicy(
                    stop=SRStopPolicy(maximum_controller_rounds=2),
                    resume=AcceptedStateResume(
                        checkpoint_path=checkpoint_path,
                        checkpoint_sha256=tampered_sha256,
                    ),
                ),
            ),
        )
