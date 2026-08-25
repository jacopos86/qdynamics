from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import resolve_problem_context
from pipelines.static_adapt.sr_snake import (
    AcceptedStateReceipt,
    AcceptedStateResume,
    AcceptedTransitionReceipt,
    AppendOnlyInsertion,
    BeamOff,
    CheckpointObservation,
    CombinatorialBatchAdmission,
    EstimatorLedgerObservation,
    ExactEDSourceReceipt,
    ExactEDStop,
    ForkLocalBeam,
    FreshStart,
    FullCombinatorialSearchWindow,
    GreedyBatchAdmission,
    PruningOff,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRObservationPolicy,
    SRRunRequest,
    SRRunResult,
    SRStopPolicy,
    SingletonAdmission,
    TrustRegionPruning,
    run_sr_snake,
)
import pipelines.static_adapt.sr_snake as sr_snake
import pipelines.static_adapt.sr_snake._controller as sr_controller
import pipelines.static_adapt.ra_adapt.runtime as ra_runtime
from pipelines.static_adapt.sr_snake._context import (
    _resolve_execution_context,
)
from pipelines.static_adapt.sr_snake._selection import _select_singleton
from pipelines.static_adapt.sr_snake._transition import (
    _AcceptedStateSnapshot,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract,
    canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract_sha256,
)


ROUTE_PROFILE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_v1"
)
ROUTE_DIGEST = (
    "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2"
)
EXPECTED_OPERATORS = (
    (
        "uccsd_ferm_lifted::uccsd_sing(alpha:0->1)::"
        "child_set[0]::legal_projected"
    ),
    (
        "uccsd_ferm_lifted::uccsd_sing(beta:2->3)::"
        "child_set[0]::legal_projected"
    ),
)
EXPECTED_GENERATORS = (
    "gen:edc1a5f152a274be",
    "gen:6758ef9dd23ce33b",
)
EXPECTED_LEDGER_SEQUENCE_IDENTITY_SHA256 = (
    "f71b5b15d4e193a889369fed4e135ec654363779e85caceabd6ad75402249d26"
)
EXPECTED_PUBLIC_EXPORTS = {
    "AcceptedRefitReceipt",
    "AcceptedStateReceipt",
    "AcceptedStateResume",
    "AcceptedTransitionReceipt",
    "AppendCommutationReducedInsertion",
    "AppendOnlyInsertion",
    "BeamOff",
    "CANONICAL_CANDIDATE_REPRESENTATION",
    "CanonicalReportingReceipt",
    "CheckpointObservation",
    "CheckpointReceipt",
    "CombinatorialBatchAdmission",
    "CombinatorialBatchAcceptedTransitionReceipt",
    "CombinatorialBatchMemberAdmissionReceipt",
    "CombinatorialBatchProposalReceipt",
    "CombinatorialBatchScientificReplayReceipt",
    "CombinatorialBatchTransitionAdmissionReceipt",
    "EstimatorAccountingReceipt",
    "EstimatorComponentsReceipt",
    "EstimatorLedgerObservation",
    "EstimatorWorkReceipt",
    "ExactEDSourceReceipt",
    "ExactEDStop",
    "ForkLocalBeam",
    "FreshStart",
    "AlwaysCommutationReducedInsertion",
    "FullCombinatorialSearchWindow",
    "GreedyBatchAdmission",
    "GreedyBatchAcceptedTransitionReceipt",
    "GreedyBatchMemberAdmissionReceipt",
    "GreedyBatchProposalReceipt",
    "GreedyBatchScientificReplayReceipt",
    "GreedyBatchTransitionAdmissionReceipt",
    "MetricPruning",
    "ObservationArtifactReceipt",
    "ObservationReceipt",
    "ParameterBlockReceipt",
    "PhaseIIIReceipt",
    "PhaseReceipt",
    "PlateauCommutationInsertion",
    "PruningOff",
    "ReferenceStateReceipt",
    "RecoverabilityPruneReceipt",
    "ResolvedExecutionReceipt",
    "ResolvedProblemReceipt",
    "RouteReceipt",
    "RuntimePauliTermReceipt",
    "SRExecutionPolicy",
    "SRMethodPolicy",
    "SRObservationPolicy",
    "SRRunRequest",
    "SRRunResult",
    "SRStopPolicy",
    "ScientificReplayReceipt",
    "SingletonAdmission",
    "StopConditionReceipt",
    "StopReceipt",
    "SupportedMetricReceipt",
    "TrustSolveReceipt",
    "TrustRegionPruning",
    "run_sr_snake",
}


def _small_hh_problem() -> Any:
    request = ProblemRequest(
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
    )
    return resolve_problem_context(
        request,
        exact_energy_impl=adapt_pipeline._exact_gs_energy_for_problem,
    )


def _two_round_request(
    root: Path,
    *,
    checkpoint_every: int,
    exact: ExactEDStop | None = None,
) -> SRRunRequest:
    return SRRunRequest(
        method=SRMethodPolicy(insertion=AppendOnlyInsertion()),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(
                maximum_controller_rounds=2,
                exact_ed_target=exact,
            )
        ),
        observation=SRObservationPolicy(
            checkpoint=CheckpointObservation(
                path=root / "current.json",
                every_controller_rounds=checkpoint_every,
            ),
            estimator_ledger=EstimatorLedgerObservation(
                path=root / "ledger.json"
            ),
        ),
    )


def _scientific_signature(result: SRRunResult) -> dict[str, Any]:
    return {
        "final_state": result.final_state.to_dict(),
        "accepted_trajectory": [
            receipt.to_dict() for receipt in result.accepted_trajectory
        ],
        "accepted_transitions": [
            receipt.to_dict() for receipt in result.accepted_transitions
        ],
        "problem": result.problem.to_dict(),
        "route": result.route.to_dict(),
        "scientific_replay": [
            receipt.to_dict() for receipt in result.scientific_replay
        ],
        "accounting": result.estimator_accounting.to_dict(),
    }


def _accepted_refit_projection_row() -> dict[str, Any]:
    return {
        "accepted_refit": {
            "policy": "supported_fs_whitened_fixed_v1",
            "supported_rank": 2,
            "final_energy": -1.25,
            "accepted_refit_invocation": {
                "config": {
                    "scope": "full_ansatz_v1",
                    "coordinate_chart": "supported_fs_whitened_fixed_v1",
                    "base_chart_policy": (
                        "expanded_runtime_projected_logical_v1"
                    ),
                    "full_ansatz": True,
                    "supported_metric": {
                        "policy": "supported_metric_whitened_eigh_v1",
                        "rank_relative_tolerance": 1.0e-10,
                        "metric_regularization": 1.0e-12,
                        "energy_regularization": 1.0e-12,
                        "max_fubini_study_step": 0.25,
                        "global_trust_kkt_residual_accuracy": 1.0e-8,
                        "global_trust_metric_distortion_budget": 1.0e-6,
                    },
                },
                "metric_query_accounting": {
                    "symmetric_metric_element_occurrences": 3,
                },
            },
        }
    }


def test_accepted_refit_projection_preserves_v1_serialized_bytes() -> None:
    receipt = ra_runtime._accepted_refit_receipt(
        _accepted_refit_projection_row()
    )
    expected = {
        "policy": "supported_fs_whitened_fixed_v1",
        "scope": "full_ansatz_v1",
        "coordinate_chart": "supported_fs_whitened_fixed_v1",
        "base_chart_policy": "expanded_runtime_projected_logical_v1",
        "full_ansatz": True,
        "supported_rank": 2,
        "final_energy": -1.25,
        "symmetric_metric_element_occurrences": 3,
        "supported_metric": {
            "policy": "supported_metric_whitened_eigh_v1",
            "rank_relative_tolerance": 1.0e-10,
            "metric_regularization": 1.0e-12,
            "energy_regularization": 1.0e-12,
            "max_fubini_study_step": 0.25,
            "global_trust_kkt_residual_accuracy": 1.0e-8,
            "global_trust_metric_distortion_budget": 1.0e-6,
        },
    }

    assert receipt.to_json().encode("utf-8") == json.dumps(
        expected,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_accepted_refit_projection_includes_v2_initialization_evidence() -> None:
    row = _accepted_refit_projection_row()
    refit = row["accepted_refit"]
    refit["accepted_refit_initialization"] = {
        "policy": "exact_applied_joint_step_guarded_v1",
        "status": "accepted",
        "guard_objective_evals": 1,
    }
    refit["accepted_refit_initialization_guard_nfev"] = 1

    receipt = ra_runtime._accepted_refit_receipt(row)

    assert receipt.initialization_policy == (
        "exact_applied_joint_step_guarded_v1"
    )
    assert receipt.initialization_status == "accepted"
    assert receipt.initialization_guard_nfev == 1
    assert receipt.to_dict()["initialization_policy"] == (
        "exact_applied_joint_step_guarded_v1"
    )


def _ledger_occurrences(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    occurrences = payload["ledger"]["occurrences"]
    assert isinstance(occurrences, list)
    assert all(isinstance(row, dict) for row in occurrences)
    return occurrences


def _ledger_sequence_identity_sha256(
    occurrences: list[dict[str, Any]],
) -> str:
    identities = [
        {
            "sequence": int(row["sequence"]),
            "primitive_id": str(row["primitive_id"]),
        }
        for row in occurrences
    ]
    return hashlib.sha256(
        json.dumps(
            identities,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_request_contract_is_exactly_three_immutable_progressive_choices() -> None:
    signature = inspect.signature(run_sr_snake)
    assert tuple(signature.parameters) == ("problem", "request")
    assert signature.parameters["request"].default is None
    assert tuple(field.name for field in fields(SRRunRequest)) == (
        "method",
        "execution",
        "observation",
    )
    assert set(sr_snake.__all__) == EXPECTED_PUBLIC_EXPORTS

    request = SRRunRequest()
    assert request == SRRunRequest(
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=50),
            resume=FreshStart(),
        ),
        observation=SRObservationPolicy(),
    )
    with pytest.raises(FrozenInstanceError):
        request.execution = SRExecutionPolicy()  # type: ignore[misc]

    payload = request.to_dict()
    assert tuple(payload) == ("method", "execution", "observation")
    assert payload == {
            "method": {
                "admission": {"kind": "singleton"},
                "beam": {"kind": "off"},
                "insertion": {"kind": "plateau_commutation"},
                "pruning": {"kind": "off"},
            },
        "execution": {
            "resume": {"kind": "fresh_start"},
            "stop": {"maximum_controller_rounds": 50},
        },
        "observation": {},
    }
    assert "exact_ed_target" not in payload["execution"]["stop"]
    assert set(payload["method"]["admission"]) == {"kind"}
    assert set(payload["method"]["pruning"]) == {"kind"}
    assert set(payload["method"]["beam"]) == {"kind"}
    assert request.to_json() == json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def test_optional_policy_and_resume_shapes_serialize_without_dormant_fields(
    tmp_path: Path,
) -> None:
    source = ExactEDSourceReceipt(
        source_id="fixture:hh-l2",
        problem_request_sha256="a" * 64,
        sector_label="half_filling_sz0",
        comparison_space_label="fixed_particle_sector",
        n_ph_max=1,
    )
    request = SRRunRequest(
        method=SRMethodPolicy(
            admission=GreedyBatchAdmission(maximum_size=3, search_window_size=None),
            pruning=TrustRegionPruning(),
            beam=ForkLocalBeam(
                live_parent_branches=3,
                admission_children_per_parent=2,
                maximum_admission_children_per_round=6,
                s_alg_weight=0.01,
            ),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(
                maximum_controller_rounds=7,
                exact_ed_target=ExactEDStop(
                    energy=-1.25,
                    absolute_tolerance=2.0e-4,
                    source=source,
                ),
            ),
            resume=AcceptedStateResume(
                checkpoint_path=tmp_path / "checkpoint.json",
                checkpoint_sha256="b" * 64,
            ),
        ),
        observation=SRObservationPolicy(
            checkpoint=CheckpointObservation(
                path=tmp_path / "current.json",
                every_controller_rounds=2,
                keep_history_tail=11,
            ),
            estimator_ledger=EstimatorLedgerObservation(
                path=tmp_path / "ledger.json"
            ),
        ),
    )
    payload = request.to_dict()
    assert payload["method"] == {
        "admission": {
            "kind": "greedy_batch",
            "maximum_size": 3,
            "search_window_size": None,
        },
        "beam": {
            "admission_children_per_parent": 2,
            "calibration_status": "uncalibrated_default",
            "kind": "fork_local",
            "live_parent_branches": 3,
            "maximum_admission_children_per_round": 6,
            "s_alg_weight": 0.01,
        },
        "insertion": {"kind": "plateau_commutation"},
        "pruning": {"kind": "trust_region"},
    }
    assert payload["execution"]["resume"] == {
        "checkpoint_path": str(tmp_path / "checkpoint.json"),
        "checkpoint_sha256": "b" * 64,
        "kind": "accepted_state_resume",
    }
    assert payload["execution"]["stop"]["maximum_controller_rounds"] == 7
    assert payload["execution"]["stop"]["exact_ed_target"]["source"] == {
        "comparison_space_label": "fixed_particle_sector",
        "n_ph_max": 1,
        "problem_request_sha256": "a" * 64,
        "sector_label": "half_filling_sz0",
        "source_id": "fixture:hh-l2",
    }
def test_greedy_batch_admission_exposes_only_ranked_window_controls() -> None:
    with pytest.raises(TypeError):
        GreedyBatchAdmission()  # type: ignore[call-arg]
    full_window = GreedyBatchAdmission(
        maximum_size=3,
        search_window_size=None,
    )
    bounded_window = GreedyBatchAdmission(
        maximum_size=5,
        search_window_size=11,
    )

    assert full_window.maximum_size == 3
    assert full_window.search_window_size is None
    assert full_window.to_dict() == {
        "kind": "greedy_batch",
        "maximum_size": 3,
        "search_window_size": None,
    }
    assert bounded_window.to_dict() == {
        "kind": "greedy_batch",
        "maximum_size": 5,
        "search_window_size": 11,
    }
    with pytest.raises(
        ValueError,
        match="search_window_size must be a positive integer",
    ):
        GreedyBatchAdmission(maximum_size=3, search_window_size=0)


def test_combinatorial_batch_admission_requires_a_window_choice() -> None:
    with pytest.raises(TypeError):
        CombinatorialBatchAdmission()  # type: ignore[call-arg]
    fixed_window = CombinatorialBatchAdmission(
        maximum_size=3,
        search_window_size=6,
    )
    scaled_window = CombinatorialBatchAdmission(maximum_size=5, search_window_size=10)
    explicit_window = CombinatorialBatchAdmission(
        maximum_size=4,
        search_window_size=7,
    )
    full_window = CombinatorialBatchAdmission(
        maximum_size=3,
        search_window_size=FullCombinatorialSearchWindow(),
    )

    assert fixed_window.maximum_size == 3
    assert fixed_window.search_window_size == 6
    assert fixed_window.resolved_search_window_size == 6
    assert fixed_window.to_dict() == {
        "kind": "combinatorial_batch",
        "maximum_size": 3,
        "search_window_size": 6,
    }
    assert scaled_window.to_dict() == {
        "kind": "combinatorial_batch",
        "maximum_size": 5,
        "search_window_size": 10,
    }
    assert explicit_window.to_dict() == {
        "kind": "combinatorial_batch",
        "maximum_size": 4,
        "search_window_size": 7,
    }
    assert isinstance(
        full_window.search_window_size,
        FullCombinatorialSearchWindow,
    )
    assert full_window.resolved_search_window_size is None
    assert full_window.to_dict() == {
        "kind": "combinatorial_batch",
        "maximum_size": 3,
        "search_window_size": None,
    }
    with pytest.raises(
        ValueError,
        match="search_window_size must be a positive integer",
    ):
        CombinatorialBatchAdmission(maximum_size=3, search_window_size=0)
    with pytest.raises(ValueError, match="ceiling of 5"):
        CombinatorialBatchAdmission(maximum_size=6, search_window_size=10)


@pytest.mark.parametrize(
    ("maximum_size", "expected_window"),
    [(1, 2), (3, 6), (5, 10)],
)
def test_direct_combinatorial_contract_derives_window_from_batch_cap(
    maximum_size: int,
    expected_window: int,
) -> None:
    derived = (
        canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract(
            maximum_size=maximum_size,
        )
    )
    explicit = (
        canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract(
            maximum_size=maximum_size,
            search_window_size=expected_window,
        )
    )

    assert derived == explicit
    assert derived["semantic_invariants"][
        "combinatorial_batch_search_window_size"
    ] == expected_window
    assert (
        canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract_sha256(
            maximum_size=maximum_size,
        )
        == canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract_sha256(
            maximum_size=maximum_size,
            search_window_size=expected_window,
        )
    )


def test_direct_combinatorial_contract_distinguishes_full_and_fixed_windows() -> None:
    full = (
        canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract(
            maximum_size=5,
            search_window_size=None,
        )
    )
    fixed = (
        canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract(
            maximum_size=5,
            search_window_size=7,
        )
    )

    assert full["semantic_invariants"][
        "combinatorial_batch_search_window_size"
    ] is None
    assert fixed["semantic_invariants"][
        "combinatorial_batch_search_window_size"
    ] == 7
    assert (
        canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract_sha256(
            maximum_size=5,
            search_window_size=None,
        )
        != canonical_sr_snake_combinatorial_batch_projected_phase3_no_overlap_trust_v1_contract_sha256(
            maximum_size=5,
            search_window_size=7,
        )
    )


def test_stop_policy_requires_a_finite_cap_and_same_problem_exact_source() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SRStopPolicy(maximum_controller_rounds=0)

    problem = _small_hh_problem()
    source = ExactEDSourceReceipt.from_problem(
        problem,
        source_id="fixture:mismatched-cutoff",
    )
    mismatched = replace(source, n_ph_max=source.n_ph_max + 1)
    request = SRRunRequest(
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(
                maximum_controller_rounds=1,
                exact_ed_target=ExactEDStop(
                    energy=-0.75,
                    absolute_tolerance=1.0e-8,
                    source=mismatched,
                ),
            )
        )
    )
    with pytest.raises(ValueError, match="n_ph_max"):
        run_sr_snake(problem, request)


def test_observation_destinations_must_be_distinct(tmp_path: Path) -> None:
    shared = tmp_path / "same.json"
    with pytest.raises(ValueError, match="destinations must differ"):
        SRObservationPolicy(
            checkpoint=CheckpointObservation(path=shared),
            estimator_ledger=EstimatorLedgerObservation(
                path=shared.parent / "." / shared.name
            ),
        )


def test_exact_stop_does_not_fire_without_an_accepted_refit() -> None:
    problem = _small_hh_problem()
    source = ExactEDSourceReceipt.from_problem(
        problem,
        source_id="fixture:round-zero-energy",
    )
    request = SRRunRequest(
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(
                maximum_controller_rounds=2,
                exact_ed_target=ExactEDStop(
                    energy=-0.75,
                    absolute_tolerance=1.0e-12,
                    source=source,
                ),
            )
        )
    )
    round_zero = _AcceptedStateSnapshot(
        controller_round=0,
        accepted_operator_ids=(),
        accepted_insertion_positions=(),
        logical_parameter_ids=(),
        logical_parameter_values=(),
        runtime_parameter_ids=(),
        runtime_parameter_values=(),
        accepted_energy=-0.75,
        accepted_state_fingerprint="round-zero",
        available_generator_ids=("generator:0",),
        selection_counts=(("generator:0", 0),),
        trust_state_identity="trust:0",
        optimizer_memory_identity="optimizer:0",
        estimator_prefix_identity="ledger:0",
    )

    receipt = sr_controller._configured_stop_receipt(
        request.execution.stop,
        round_zero,
    )

    assert receipt.primary_reason == "controller_continues"
    assert receipt.fired_reasons == ()
    assert receipt.conditions[0].reason == "maximum_controller_rounds"
    assert receipt.conditions[0].fired is False
    assert receipt.conditions[1].reason == "exact_ed_target_reached"
    assert receipt.conditions[1].fired is False
    assert receipt.exact_observed_absolute_difference == 0.0


@pytest.mark.parametrize(
    "components, s_alg, message",
    [
        (
            {
                "N_H_outer": 1,
                "N_H_refit": 2,
                "N_grad": 3,
            },
            6,
            "N_metric",
        ),
        (
            {
                "N_H_outer": 1,
                "N_H_refit": 2,
                "N_grad": 3,
                "N_metric": 4,
            },
            11,
            "does not equal",
        ),
    ],
)
def test_estimator_work_requires_complete_reconciled_components(
    components: dict[str, int],
    s_alg: int,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        ra_runtime._work({"components": components, "S_alg": s_alg})


@pytest.mark.parametrize(
    "typed_request",
    [
        SRRunRequest(
            method=SRMethodPolicy(beam=ForkLocalBeam())
        ),
        SRRunRequest(
            method=SRMethodPolicy(
                admission=CombinatorialBatchAdmission(maximum_size=3, search_window_size=6),
                beam=ForkLocalBeam(),
            )
        ),
    ],
)
def test_composition_policies_reach_the_canonical_context(
    typed_request: SRRunRequest,
) -> None:
    context = _resolve_execution_context(
        _small_hh_problem(),
        typed_request,
    )

    assert context.request.method == typed_request.method
    assert context.request.method.beam.kind == "fork_local"


def test_resume_fails_on_a_missing_authenticated_checkpoint() -> None:
    with pytest.raises(
        ValueError,
        match="regular, non-symlink file",
    ):
        run_sr_snake(
            _small_hh_problem(),
            SRRunRequest(
                execution=SRExecutionPolicy(
                    stop=SRStopPolicy(maximum_controller_rounds=1),
                    resume=AcceptedStateResume(
                        checkpoint_path=Path("checkpoint.json"),
                        checkpoint_sha256="c" * 64,
                    ),
                )
            ),
        )


def test_default_facade_does_not_call_either_legacy_loop_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)

    def _legacy_forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the default facade entered a legacy loop")

    monkeypatch.setattr(
        adapt_pipeline,
        "_run_hardcoded_adapt_vqe",
        _legacy_forbidden,
    )
    assert not hasattr(
        adapt_pipeline,
        "_run_hardcoded_adapt_vqe_program",
    )

    result = run_sr_snake(
        _small_hh_problem(),
        SRRunRequest(
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            )
        ),
    )

    assert result.stop.completed_controller_rounds == 1
    assert len(result.accepted_transitions) == 1


def test_default_transition_dispatches_geometry_expansion_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)

    geometry_calls = 0
    ordinary_calls = 0
    joint_step_calls = 0
    original_geometry_update = (
        adapt_pipeline.update_geometry_expansion_trust_region_state
    )
    original_ordinary_update = adapt_pipeline.update_trust_region_state
    original_joint_step = (
        adapt_pipeline._accepted_sr_v2_joint_coordinate_step
    )

    def _force_geometry_expansion(
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], str, bool, bool]:
        return (
            {},
            adapt_pipeline.HISTORICAL_SINGLETON_GEOMETRY_EXPANSION_CONTEXT_V1,
            False,
            True,
        )

    def _record_geometry_update(*args: Any, **kwargs: Any) -> Any:
        nonlocal geometry_calls
        geometry_calls += 1
        return original_geometry_update(*args, **kwargs)

    def _record_ordinary_update(*args: Any, **kwargs: Any) -> Any:
        nonlocal ordinary_calls
        ordinary_calls += 1
        return original_ordinary_update(*args, **kwargs)

    def _record_joint_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal joint_step_calls
        joint_step_calls += 1
        return original_joint_step(*args, **kwargs)

    monkeypatch.setattr(
        adapt_pipeline,
        "_historical_singleton_trust_update_inputs_or_geometry_expansion",
        _force_geometry_expansion,
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "update_geometry_expansion_trust_region_state",
        _record_geometry_update,
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "update_trust_region_state",
        _record_ordinary_update,
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "_accepted_sr_v2_joint_coordinate_step",
        _record_joint_step,
    )

    result = run_sr_snake(
        _small_hh_problem(),
        SRRunRequest(
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            )
        ),
    )

    assert result.stop.completed_controller_rounds == 1
    assert geometry_calls == 1
    assert ordinary_calls == 0
    assert joint_step_calls == 0
    trust = result.scientific_replay[-1].trust_solve
    assert trust.transaction_complete is None
    assert trust.transaction_failure == (
        adapt_pipeline.GEOMETRY_EXPANSION_SOURCE_METRIC_LIMITATION
    )
    assert trust.endpoint_overlap_query_charge == 0


def test_default_numerical_session_has_explicit_non_coroutine_state() -> None:
    context_fields = {
        item.name
        for item in fields(adapt_pipeline._DefaultNoPruneKernelContext)
    }
    cursor_fields = {
        item.name
        for item in fields(adapt_pipeline._DefaultNoPruneNumericalCursor)
    }
    session_fields = {
        item.name
        for item in fields(adapt_pipeline._DefaultNoPruneNumericalSession)
    }

    assert {
        "hamiltonian",
        "compiled_hamiltonian",
        "pool",
        "reference_state",
        "route_profile",
        "route_contract_sha256",
    } <= context_fields
    assert {
        "controller_round",
        "selected_ops",
        "theta",
        "selected_layout",
        "selected_executor",
        "accepted_energy",
        "history",
        "phase2_optimizer_memory",
        "route_a_trust_region_state",
    } <= cursor_fields
    assert "program" not in session_fields
    assert "generator" not in session_fields
    assert not inspect.isgeneratorfunction(
        adapt_pipeline._run_hardcoded_adapt_vqe
    )
    for method_name in (
        "prepare_selection",
        "prepare_transition",
        "project_accepted_event",
        "finalize",
        "close",
    ):
        method = getattr(
            adapt_pipeline._DefaultNoPruneNumericalSession,
            method_name,
        )
        assert not inspect.isgeneratorfunction(method)
        assert not inspect.iscoroutinefunction(method)


def test_direct_transition_starts_from_the_exact_prepared_selection_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    context = _resolve_execution_context(
        _small_hh_problem(),
        SRRunRequest(
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            )
        ),
    )
    runtime = context.build_default_controller_runtime()

    try:
        prepared = runtime.prepare_selection(runtime.initial_accepted_state)
        decision = _select_singleton(
            prepared.controller_state,
            prepared.workspace,
        )
        workspace = runtime.prepare_transition(
            runtime.initial_accepted_state,
            decision,
        )

        assert (
            workspace.numerical_runtime.accepted_state_snapshot()
            == runtime.initial_accepted_state
        )
    finally:
        runtime.close()


def test_facade_reproduces_issue7_and_exact_stop_composes_after_accepted_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    problem = _small_hh_problem()

    baseline = run_sr_snake(
        problem,
        _two_round_request(
            tmp_path / "baseline",
            checkpoint_every=1,
        ),
    )
    baseline_checkpoint_path = tmp_path / "baseline" / "current.json"
    baseline_checkpoint = json.loads(
        baseline_checkpoint_path.read_text(encoding="utf-8")
    )
    baseline_adapt = baseline_checkpoint["adapt_vqe"]
    assert {
        path.name
        for path in (tmp_path / "baseline").glob(
            "current.estimator_call_ledger_checkpoint.*.json"
        )
    } == {
        baseline_adapt["estimator_call_ledger_checkpoint"]["path"]
    }
    assert {
        path.name
        for path in (tmp_path / "baseline").glob(
            "current.verified_singleton_resume.*.json"
        )
    } == {
        baseline_adapt["verified_singleton_resume_sidecar"]["path"]
    }
    baseline_occurrences = _ledger_occurrences(
        tmp_path / "baseline" / "ledger.json"
    )
    assert len(baseline_occurrences) == 709
    assert tuple(
        int(row["sequence"]) for row in baseline_occurrences
    ) == tuple(range(1, 710))
    assert (
        _ledger_sequence_identity_sha256(baseline_occurrences)
        == EXPECTED_LEDGER_SEQUENCE_IDENTITY_SHA256
    )
    assert isinstance(baseline, SRRunResult)
    assert all(
        isinstance(receipt, AcceptedTransitionReceipt)
        for receipt in baseline.accepted_transitions
    )
    assert len(baseline.accepted_transitions) == 2
    assert tuple(
        receipt.controller_round
        for receipt in baseline.accepted_transitions
    ) == (1, 2)
    assert tuple(
        receipt.generator_id for receipt in baseline.accepted_transitions
    ) == EXPECTED_GENERATORS
    assert tuple(
        receipt.cumulative_s_alg
        for receipt in baseline.accepted_transitions
    ) == (299, 709)
    assert tuple(
        receipt.cumulative_s_unique
        for receipt in baseline.accepted_transitions
    ) == (250, 564)
    assert baseline.final_state.operators == EXPECTED_OPERATORS
    assert baseline.final_state.insertion_positions == (0, 1)
    assert baseline.final_state.energy == pytest.approx(
        -0.749999999999968,
        abs=1.0e-10,
    )
    assert tuple(
        receipt.generator_id for receipt in baseline.scientific_replay
    ) == EXPECTED_GENERATORS
    assert tuple(
        receipt.accepted_state.energy
        for receipt in baseline.scientific_replay
    ) == pytest.approx(
        (0.2192235935955847, -0.749999999999968),
        abs=1.0e-10,
    )
    assert tuple(
        receipt.phase.phase3.coordinate_indices
        for receipt in baseline.scientific_replay
    ) == ((0,), (0, 1))
    assert tuple(
        receipt.phase.phase3.supported_rank
        for receipt in baseline.scientific_replay
    ) == (1, 2)
    assert baseline.accepted_trajectory[0].logical_parameters == pytest.approx(
        (0.9078874980073425,),
        abs=1.0e-10,
    )
    assert baseline.accepted_trajectory[1].logical_parameters == pytest.approx(
        (0.7853981672776273, 0.7853980367776333),
        abs=1.0e-10,
    )
    assert baseline.scientific_replay[0].trust_solve.transaction_complete is True
    assert baseline.scientific_replay[0].trust_solve.transaction_failure is None
    assert baseline.scientific_replay[1].trust_solve.transaction_complete is True
    assert baseline.scientific_replay[1].trust_solve.supported_rank == 2
    assert tuple(
        receipt.accepted_refit.supported_rank
        for receipt in baseline.scientific_replay
    ) == (1, 2)
    assert tuple(
        receipt.accepted_refit.symmetric_metric_element_occurrences
        for receipt in baseline.scientific_replay
    ) == (1, 3)
    assert all(
        receipt.accepted_refit.supported_metric.policy
        == "supported_metric_whitened_eigh_v1"
        for receipt in baseline.scientific_replay
    )
    assert all(
        receipt.checkpoint.strict_replay_passed
        for receipt in baseline.scientific_replay
    )
    assert tuple(
        tuple(
            term.pauli_exyz
            for block in receipt.checkpoint.parameter_blocks
            for term in block.runtime_terms
        )
        for receipt in baseline.scientific_replay
    ) == (("eeeeyx",), ("eeeeyx", "eeyxee"))
    assert tuple(
        receipt.checkpoint.ordered_operator_labels
        for receipt in baseline.scientific_replay
    ) == (EXPECTED_OPERATORS[:1], EXPECTED_OPERATORS)
    assert tuple(
        receipt.checkpoint.checkpoint_sha256
        for receipt in baseline.scientific_replay
    ) == (
        "aeebb5822abd94dbb03654a2e0c05e7ae74e4c5f654684340d864374c2246cd4",
        "567b28929892422330eae2de2dd605640aa892f5ce19edf188faf8fd8f115efa",
    )
    assert tuple(
        receipt.checkpoint.estimator_ledger_s_alg
        for receipt in baseline.scientific_replay
    ) == (299, 709)
    assert baseline.route.family == "singleton_response_snake"
    assert baseline.route.profile_request == (
        "sr_snake_no_prune_symmetric_cost_projected_phase3_"
        "no_overlap_trust_v1"
    )
    assert baseline.route.profile == ROUTE_PROFILE
    assert baseline.route.contract_sha256 == ROUTE_DIGEST
    assert baseline.route.execution.pool == "full_meta"
    assert baseline.route.execution.optimizer == "POWELL"
    assert baseline.route.execution.seed == 7
    assert baseline.route.execution.phase0_enabled is False
    assert baseline.route.execution.phase2_batching_enabled is False
    assert baseline.route.execution.phase3_batching_enabled is False
    assert baseline.route.execution.pruning_enabled is False
    assert baseline.route.execution.beam_enabled is False
    assert baseline.route.execution.phase_live_hysteresis_enabled is False
    assert baseline.estimator_accounting.complete is True
    assert baseline.estimator_accounting.all_work.s_alg == 709
    assert baseline.estimator_accounting.winning_lineage.s_alg == 709
    assert baseline.estimator_accounting.all_work.components.to_dict() == {
        "n_h_outer": 2,
        "n_h_refit": 124,
        "n_grad": 251,
        "n_metric": 332,
    }
    assert baseline.estimator_accounting.raw_occurrence_total == 709
    assert baseline.estimator_accounting.prefix_closure_passed is True
    assert baseline.stop.primary_reason == "maximum_controller_rounds"
    assert baseline.stop.completed_controller_rounds == 2
    assert baseline.stop.accepted_operator_count == 2
    assert baseline.stop.fired_reasons == ("maximum_controller_rounds",)
    assert set(baseline.to_dict()) == {
        "final_state",
        "accepted_trajectory",
        "accepted_transitions",
        "problem",
        "route",
        "stop",
        "scientific_replay",
            "estimator_accounting",
            "observation",
            "canonical_reporting",
        }
    assert json.loads(baseline.to_json()) == baseline.to_dict()

    source = ExactEDSourceReceipt.from_problem(
        problem,
        source_id="fixture:issue7-final-energy",
    )

    def _unexpected_exact_solver(**_kwargs: Any) -> float:
        raise AssertionError("predefined exact stop must not invoke exact ED")

    no_solver_problem = replace(
        problem,
        exact_target=replace(
            problem.exact_target,
            resolve_energy=_unexpected_exact_solver,
        ),
    )
    observed = run_sr_snake(
        problem,
        _two_round_request(
            tmp_path / "observed",
            checkpoint_every=2,
        ),
    )
    observed_occurrences = _ledger_occurrences(
        tmp_path / "observed" / "ledger.json"
    )
    assert observed_occurrences == baseline_occurrences
    assert _scientific_signature(observed) == _scientific_signature(baseline)
    assert baseline.observation != observed.observation
    assert {receipt.kind for receipt in baseline.observation.artifacts} == {
        "accepted_state_checkpoint",
        "estimator_ledger",
    }
    assert all(
        receipt.sha256 and receipt.size_bytes > 0
        for receipt in baseline.observation.artifacts
    )

    exact = ExactEDStop(
        energy=baseline.final_state.energy,
        absolute_tolerance=1.0e-10,
        source=source,
    )
    exact_result = run_sr_snake(
        no_solver_problem,
        _two_round_request(
            tmp_path / "exact",
            checkpoint_every=2,
            exact=exact,
        ),
    )
    assert exact_result.final_state == baseline.final_state
    assert exact_result.accepted_trajectory == baseline.accepted_trajectory
    assert exact_result.route == baseline.route
    assert exact_result.estimator_accounting.all_work == (
        baseline.estimator_accounting.all_work
    )
    assert exact_result.estimator_accounting.winning_lineage == (
        baseline.estimator_accounting.winning_lineage
    )
    assert exact_result.stop.primary_reason == "exact_ed_target_reached"
    assert exact_result.stop.fired_reasons == (
        "exact_ed_target_reached",
        "maximum_controller_rounds",
    )
    assert exact_result.stop.completed_controller_rounds == 2
    assert exact_result.stop.exact_observed_absolute_difference == pytest.approx(
        0.0,
        abs=1.0e-15,
    )
    assert exact_result.stop.exact_source == source

    initial_accepted_energy = baseline.scientific_replay[0].energy_before_refit
    accepted_only = run_sr_snake(
        no_solver_problem,
        SRRunRequest(
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(
                    maximum_controller_rounds=1,
                    exact_ed_target=ExactEDStop(
                        energy=initial_accepted_energy,
                        absolute_tolerance=1.0e-12,
                        source=source,
                    ),
                )
            )
        ),
    )
    assert accepted_only.stop.completed_controller_rounds == 1
    assert accepted_only.stop.primary_reason == "maximum_controller_rounds"
    assert accepted_only.stop.fired_reasons == ("maximum_controller_rounds",)
    assert accepted_only.stop.exact_observed_absolute_difference is not None
    assert accepted_only.stop.exact_observed_absolute_difference > 1.0e-6
