from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction

import pytest

from pipelines.static_adapt.sr_snake_escape_controller import (
    reachable_population_digest,
)
from pipelines.static_adapt.sr_snake_modeled_minimum import (
    ACTION_INDEX_SCHEMA,
    EligibilityStateToken,
    EnergyInterval,
    FrozenServiceItem,
    LogEntitlement,
    PathActionKey,
    PathOrientation,
    RunEnergyUnit,
    ServiceTag,
)
from pipelines.static_adapt.sr_snake_modeled_minimum_runtime import (
    ConfigurationBinding,
    ExactComplexCoefficient,
    ExactOperatorPayload,
    ExactOperatorTerm,
    ExactThetaVector,
    ExecutionHistoryEvent,
    OperatorExecutionMode,
    ParameterizationMode,
    ParameterLayout,
    PreparedStateManifest,
    ProviderIdentity,
    ProviderRole,
    ReplayableStatePayload,
    ReplayableStateSnapshot,
    ServiceCursor,
    SingleBranchState,
    SourceBinding,
    StageBActionServicePlan,
    StageBExecutionCheckpoint,
    StageBExecutionState,
    StageBProviderBindings,
    StateLineageEvent,
    StrictReplayReceipt,
    ThetaSpace,
    assess_stage_b_readiness,
    external_incumbent_view,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _provider(role: ProviderRole, suffix: str = "v1") -> ProviderIdentity:
    return ProviderIdentity(
        role=role,
        provider_id=f"{role.value}:{suffix}",
        version=suffix,
        implementation_digest=_sha(f"provider:{role.value}:{suffix}"),
    )


def _source(suffix: str = "v1") -> SourceBinding:
    return SourceBinding(
        repository_id="holstein",
        revision=f"revision-{suffix}",
        source_digest=_sha(f"source:{suffix}"),
    )


def _config(suffix: str = "v1") -> ConfigurationBinding:
    return ConfigurationBinding(
        config_id=f"stage-b-{suffix}",
        route_family="singleton_response_snake",
        route_profile="supported_whitened_adaptive_trust_saddle_modeled_minimum_escape_v2",
        config_digest=_sha(f"config:{suffix}"),
        state_replay_tolerance=1.0e-10,
        state_norm_error_tolerance=1.0e-10,
    )


def _providers(*, omit: ProviderRole | None = None, suffix: str = "v1") -> StageBProviderBindings:
    def selected(role: ProviderRole) -> ProviderIdentity | None:
        return None if role is omit else _provider(role, suffix)

    return StageBProviderBindings(
        canonical_path=selected(ProviderRole.CANONICAL_PATH),
        uniform_incumbent_barrier=selected(ProviderRole.UNIFORM_INCUMBENT_BARRIER),
        nonlinear_active_manifold_distance=selected(
            ProviderRole.NONLINEAR_ACTIVE_MANIFOLD_DISTANCE
        ),
        connected_component_refit=selected(
            ProviderRole.CONNECTED_COMPONENT_REFIT
        ),
        disposable_powell=selected(ProviderRole.DISPOSABLE_POWELL),
        state_replay=selected(ProviderRole.STATE_REPLAY),
    )


def _snapshot(
    state_id: str,
    theta: float,
    *,
    source: SourceBinding,
    config: ConfigurationBinding,
    replay: bool = True,
    parent: str | None = None,
) -> ReplayableStateSnapshot:
    operator = ExactOperatorPayload(
        operator_id=f"op:{state_id}",
        semantic_operator_id="pool:x0",
        execution_mode=OperatorExecutionMode.TERMWISE_PRODUCT,
        terms=(
            ExactOperatorTerm(
                term_id="term:0",
                pauli_word="x",
                coefficient=ExactComplexCoefficient.from_complex(1.0 + 0.0j),
            ),
        ),
    )
    runtime_theta = ExactThetaVector.from_floats(
        space=ThetaSpace.RUNTIME,
        parameter_ids=("runtime:0",),
        values=(theta,),
    )
    logical_theta = ExactThetaVector.from_floats(
        space=ThetaSpace.LOGICAL,
        parameter_ids=("logical:0",),
        values=(theta,),
    )
    layout = ParameterLayout(
        runtime_parameter_ids=("runtime:0",),
        logical_parameter_ids=("logical:0",),
        runtime_to_logical=(0,),
        operator_to_logical=(0,),
    )
    prepared = PreparedStateManifest(
        state_fingerprint=state_id,
        prepared_state_digest=_sha(f"prepared:{state_id}"),
        statevector_digest=_sha(f"statevector:{state_id}"),
        preparation_manifest_digest=_sha(f"manifest:{state_id}"),
        qubit_count=1,
        normalized=True,
        finite=True,
        norm_error_bound=1.0e-12,
        phase_convention="first_nonzero_real_positive",
    )
    payload = ReplayableStatePayload(
        operators=(operator,),
        parameterization_mode=ParameterizationMode.LOGICAL_SHARED,
        runtime_theta=runtime_theta,
        logical_theta=logical_theta,
        layout=layout,
        prepared_state=prepared,
    )
    energy = EnergyInterval(
        state_id=state_id,
        energy_estimate=-1.0 + theta / 10.0,
        energy_error_bound=1.0e-10,
        comparison_epoch="comparison:1",
        simultaneous=True,
    )
    receipt = (
        StrictReplayReceipt.record_verified_result(
            receipt_id=f"replay:{state_id}",
            replay_provider=_provider(ProviderRole.STATE_REPLAY),
            source_digest=source.source_digest,
            config_digest=config.config_digest,
            payload=payload,
            energy=energy,
            projective_distance=1.0e-13,
            state_consistency_tolerance=1.0e-10,
            verification_result_digest=_sha(f"verification:{state_id}"),
            finite=True,
            normalized=True,
            phase_aligned=True,
        )
        if replay
        else None
    )
    lineage = (
        StateLineageEvent(
            event_index=0,
            event_kind="initial" if parent is None else "modeled_minimum_move",
            state_fingerprint=state_id,
            parent_state_fingerprint=parent,
            details_digest=_sha(f"lineage:{state_id}"),
            action_receipt_digest=None,
        ),
    )
    return ReplayableStateSnapshot(
        payload=payload,
        energy=energy,
        lineage=lineage,
        replay_receipt=receipt,
    )


def _token(working_state_id: str) -> EligibilityStateToken:
    records = ("record:0",)
    return EligibilityStateToken(
        working_state_fingerprint=working_state_id,
        reachable_record_ids=records,
        reachable_population_digest=reachable_population_digest(records),
        comparison_epoch="comparison:1",
        support_provenance_digest=_sha("support"),
        trust_provenance_digest=_sha("trust"),
        trust_radius=0.25,
        stationarity_margin=-1.0e-8,
    )


def _state(
    *,
    omit_provider: ProviderRole | None = None,
    replay_incumbent: bool = True,
    replay_working: bool = True,
) -> StageBExecutionState:
    source = _source()
    config = _config()
    providers = _providers(omit=omit_provider)
    incumbent = _snapshot(
        "state:I",
        0.1,
        source=source,
        config=config,
        replay=replay_incumbent,
    )
    working = _snapshot(
        "state:X",
        0.2,
        source=source,
        config=config,
        replay=replay_working,
        parent="state:I",
    )
    token = _token(working.state_fingerprint)
    energy_unit = RunEnergyUnit(run_id="run:1", unit_id="hartree", value=1.0)
    key = PathActionKey(
        record_id="record:0",
        record_order=1,
        record_count=1,
        orientation=PathOrientation.NEGATIVE,
        radius_index=2,
        path_index=3,
    )
    plan = StageBActionServicePlan.create(
        plan_id="plan:0",
        action_key=key,
        eligibility_token=token,
        energy_unit=energy_unit,
        incumbent=incumbent,
        working=working,
        source=source,
        config=config,
        providers=providers,
        service_epoch="service:1",
        service_ordinal=0,
    )
    branch = SingleBranchState(
        branch_id="branch:0",
        incumbent=incumbent,
        working=working,
        chi=0.15,
    )
    history = (
        ExecutionHistoryEvent(
            event_index=0,
            event_kind="working_state_preserved",
            incumbent_snapshot_digest=incumbent.content_digest,
            working_snapshot_digest=working.content_digest,
            chi=0.15,
            rho=0.25,
            completed_services=4,
            next_action_index=key.action_index + 1,
            details_digest=_sha("execution-history:0"),
        ),
    )
    service_population = (
        FrozenServiceItem(
            tag=ServiceTag.REFINEMENT,
            action_key=key,
            frozen_entitlement=LogEntitlement.from_coefficient(
                Fraction(1, 2), symbolic_expression="1/(2*pi^2)"
            ),
            service_count=4,
            service_epoch="service:1",
            eligibility_token_digest=token.digest,
            energy_unit_digest=energy_unit.digest,
            activation_reason="fixture_unresolved",
        ),
    )
    return StageBExecutionState(
        branch=branch,
        rho=0.25,
        core_token=token,
        energy_unit=energy_unit,
        queue=(plan,),
        service_population=service_population,
        cursor=ServiceCursor(
            service_epoch="service:1",
            next_action_index=key.action_index + 1,
            expansion_count=key.action_index,
            completed_services=4,
        ),
        providers=providers,
        source=source,
        config=config,
        history=history,
    )


def test_dual_incumbent_working_checkpoint_roundtrip_is_exact() -> None:
    state = _state()
    checkpoint = StageBExecutionCheckpoint.create_replay_complete(state)

    restored = StageBExecutionCheckpoint.from_json(
        checkpoint.to_json(),
        expected_source=state.source,
        expected_config=state.config,
        expected_providers=state.providers,
        require_replay_complete=True,
    )

    assert restored == checkpoint
    assert restored.state.branch.incumbent.state_fingerprint == "state:I"
    assert restored.state.branch.working.state_fingerprint == "state:X"
    assert restored.state.branch.exploring is True
    assert restored.state.branch.chi == 0.15
    assert restored.state.rho == 0.25
    assert restored.runtime_resume_complete is True
    assert restored.state.combined_execution_enabled is False
    assert restored.combined_execution_enabled is False


def test_top_level_and_external_views_remain_incumbent_when_x_differs() -> None:
    state = _state()
    checkpoint = StageBExecutionCheckpoint.create_replay_complete(state)

    assert state.top_level_incumbent.state_fingerprint == "state:I"
    assert state.branch.working.state_fingerprint == "state:X"

    state_view = external_incumbent_view(state)
    checkpoint_view = external_incumbent_view(checkpoint)
    assert state_view.incumbent.state_fingerprint == "state:I"
    assert checkpoint_view.incumbent.state_fingerprint == "state:I"
    assert state_view.energy == state.branch.incumbent.energy
    assert "working" not in state_view.to_dict()
    assert "state:X" not in json.dumps(state_view.to_dict(), sort_keys=True)


def test_checkpoint_rejects_payload_tamper_and_binding_drift() -> None:
    state = _state()
    checkpoint = StageBExecutionCheckpoint.create_replay_complete(state)
    data = json.loads(checkpoint.to_json())
    data["state"]["branch"]["working"]["energy"]["energy_estimate"] += 0.5

    with pytest.raises(ValueError, match="checkpoint state is invalid"):
        StageBExecutionCheckpoint.from_json(json.dumps(data))

    with pytest.raises(ValueError, match="source binding drift"):
        StageBExecutionCheckpoint.from_json(
            checkpoint.to_json(), expected_source=_source("drift")
        )
    with pytest.raises(ValueError, match="config binding drift"):
        StageBExecutionCheckpoint.from_json(
            checkpoint.to_json(), expected_config=_config("drift")
        )
    with pytest.raises(ValueError, match="provider binding drift"):
        StageBExecutionCheckpoint.from_json(
            checkpoint.to_json(), expected_providers=_providers(suffix="drift")
        )
    with pytest.raises(ValueError, match="action-index schema drift"):
        StageBExecutionCheckpoint.from_json(
            checkpoint.to_json(), expected_action_index_schema="obsolete"
        )


def test_action_schema_tamper_rejected_even_if_outer_digest_is_recomputed() -> None:
    checkpoint = StageBExecutionCheckpoint.create_replay_complete(_state())
    data = json.loads(checkpoint.to_json())
    data["action_index_schema"] = "obsolete"
    unsigned = {key: value for key, value in data.items() if key != "content_digest"}
    data["content_digest"] = hashlib.sha256(
        json.dumps(
            unsigned,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="action-index schema drift"):
        StageBExecutionCheckpoint.from_json(json.dumps(data))


def test_missing_provider_or_replay_evidence_keeps_readiness_false() -> None:
    provider_missing = _state(omit_provider=ProviderRole.DISPOSABLE_POWELL)
    provider_audit = assess_stage_b_readiness(provider_missing)
    assert provider_audit.runtime_resume_complete is False
    assert "provider_missing:disposable_powell" in provider_audit.blockers
    assert provider_audit.providers_complete is False
    assert StageBExecutionCheckpoint.create(provider_missing).runtime_resume_complete is False

    replay_missing = _state(replay_working=False)
    replay_audit = assess_stage_b_readiness(replay_missing)
    assert replay_audit.runtime_resume_complete is False
    assert "working_strict_replay_missing_or_failed" in replay_audit.blockers
    with pytest.raises(ValueError, match="not replay-complete"):
        StageBExecutionCheckpoint.create_replay_complete(replay_missing)


def test_i_only_payload_never_falls_back_working_state_to_incumbent() -> None:
    state = _state()
    data = state.to_dict()
    del data["branch"]["working"]

    with pytest.raises(ValueError, match="I-only fallback is forbidden"):
        StageBExecutionState.from_dict(data)


def test_pure_core_checkpoint_scope_is_never_replay_complete() -> None:
    payload = json.dumps(
        {
            "schema_version": "sr_snake_modeled_minimum_core_scheduler_checkpoint_v2",
            "checkpoint_scope": "pure_core_scheduler_only",
            "action_index_schema": ACTION_INDEX_SCHEMA,
            "runtime_resume_complete": False,
        }
    )

    with pytest.raises(ValueError, match="pure-core scheduler checkpoint"):
        StageBExecutionCheckpoint.from_json(payload)


def test_runtime_envelope_keeps_integration_and_combined_execution_disabled() -> None:
    checkpoint = StageBExecutionCheckpoint.create_replay_complete(_state())

    assert checkpoint.runtime_resume_complete is True
    assert checkpoint.integration_ready is False
    assert checkpoint.combined_execution_enabled is False
    assert checkpoint.readiness.integration_ready is False
    assert checkpoint.state.integration_ready is False


def test_arbitrary_size_countable_action_cursor_roundtrips_as_hex() -> None:
    state = _state()
    huge = 1 << 20000
    cursor = ServiceCursor(
        service_epoch=state.cursor.service_epoch,
        next_action_index=huge + 1,
        expansion_count=huge,
        completed_services=state.cursor.completed_services,
    )
    history = (
        replace(state.history[-1], next_action_index=huge + 1),
    )
    enlarged = replace(state, cursor=cursor, history=history)

    checkpoint = StageBExecutionCheckpoint.create_replay_complete(enlarged)
    restored = StageBExecutionCheckpoint.from_json(
        checkpoint.to_json(), require_replay_complete=True
    )

    assert restored.state.cursor.next_action_index == huge + 1
    assert "0x1" in checkpoint.to_json()


def test_theta_projection_and_generator_semantics_fail_closed() -> None:
    state = _state()
    payload = state.branch.working.payload
    inconsistent_logical = ExactThetaVector.from_floats(
        space=ThetaSpace.LOGICAL,
        parameter_ids=payload.logical_theta.parameter_ids,
        values=(999.0,),
    )

    with pytest.raises(ValueError, match="block-mean projection"):
        replace(payload, logical_theta=inconsistent_logical)

    with pytest.raises(ValueError, match="real Pauli coefficients"):
        ExactOperatorPayload(
            operator_id="bad",
            semantic_operator_id="bad",
            execution_mode=OperatorExecutionMode.TERMWISE_PRODUCT,
            terms=(
                ExactOperatorTerm(
                    term_id="bad:0",
                    pauli_word="x",
                    coefficient=ExactComplexCoefficient.from_complex(1.0j),
                ),
            ),
        )


def test_replay_provider_tolerance_and_terminal_history_are_bound() -> None:
    state = _state()
    drift_provider = _provider(ProviderRole.STATE_REPLAY, "drift")
    drift_receipt = replace(
        state.branch.working.replay_receipt,
        replay_provider=drift_provider,
    )
    drift_working = replace(
        state.branch.working,
        replay_receipt=drift_receipt,
    )
    drift_branch = replace(state.branch, working=drift_working)
    drift_plans = tuple(
        replace(plan, working_snapshot_digest=drift_working.content_digest)
        for plan in state.queue
    )
    drift_history = (
        replace(
            state.history[-1],
            working_snapshot_digest=drift_working.content_digest,
        ),
    )
    drift_state = replace(
        state,
        branch=drift_branch,
        queue=drift_plans,
        history=drift_history,
    )
    audit = assess_stage_b_readiness(drift_state)
    assert audit.runtime_resume_complete is False
    assert "working_replay_provider_binding_mismatch" in audit.blockers

    loose_config = replace(state.config, state_replay_tolerance=1.0e-6)
    loose_plans = tuple(
        replace(plan, config_binding_digest=loose_config.content_digest)
        for plan in state.queue
    )
    loose_state = replace(state, config=loose_config, queue=loose_plans)
    audit = assess_stage_b_readiness(loose_state)
    assert "incumbent_replay_tolerance_binding_mismatch" in audit.blockers
    assert "working_replay_tolerance_binding_mismatch" in audit.blockers

    with pytest.raises(ValueError, match="terminal history"):
        replace(
            state,
            history=(replace(state.history[-1], completed_services=3),),
        )


def test_state_replay_binding_is_required() -> None:
    state = _state(omit_provider=ProviderRole.STATE_REPLAY)
    audit = assess_stage_b_readiness(state)

    assert audit.runtime_resume_complete is False
    assert "provider_missing:state_replay" in audit.blockers
    assert "incumbent_replay_provider_binding_mismatch" in audit.blockers
    assert "working_replay_provider_binding_mismatch" in audit.blockers
    OperatorExecutionMode,
    ParameterizationMode,
