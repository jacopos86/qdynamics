from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pipelines.contracts.problem import (
    ExactTargetSpec,
    HamiltonianFamilyCapabilities,
    ProblemRequest,
    ReferenceStateSpec,
    RegisterLayoutSpec,
    ResolvedProblemContext,
    SectorSelection,
)
from pipelines.static_adapt.ra_adapt import append
from pipelines.static_adapt.ra_adapt import bundles as bundle_module
from pipelines.static_adapt.ra_adapt.adapters import (
    MacroCandidateAdapter,
    SinglePauliWordCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_STATIONARY,
    APPEND_ADAPT_PROTOCOL_SCHEMA,
    APPEND_CONVENTIONAL_SELECTOR_ID,
    APPEND_CONVENTIONAL_SELECTOR_SCOPE,
    AppendAdaptRequest,
    BundleProtocolMaterializationAuthority,
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART,
    NATIVE_REFIT_CHART,
    PoolInventoryReceipt,
    RESOURCE_WEIGHTING_LATE,
    canonical_json_bytes,
    canonical_sha256,
    resolved_ra_adapt_protocol_from_mapping,
)
from pipelines.reporting.paper_i_append_run_summary import (
    PAPER_I_APPEND_RUN_SUMMARY_SCHEMA,
    summarize_paper_i_append_run,
)
from pipelines.static_adapt.ra_adapt.pools import (
    CandidateInventory,
    CandidateRecord,
    _receipt,
)
from pipelines.static_adapt.ra_adapt.replay_evidence import (
    build_append_controller_replay_evidence,
    build_signed_append_prefix_checkpoint,
    validate_controller_replay_evidence,
)
from pipelines.static_adapt.numerical_physical_integrity import (
    build_append_numerical_physical_integrity,
)
from pipelines.static_adapt.sr_snake.contracts import (
    CheckpointObservation,
    SRExecutionPolicy,
    SRObservationPolicy,
    SRStopPolicy,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _problem(*, family: str = "hh", num_sites: int = 2) -> ResolvedProblemContext:
    request = ProblemRequest(
        problem_key=family,
        num_sites=num_sites,
        t=1.0,
        u=8.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.353553390593,
        n_ph_max=3,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    return ResolvedProblemContext(
        family_key=family,
        request=request,
        layout=RegisterLayoutSpec(
            total_qubits=2,
            fermion_qubits=2,
            boson_qubits=0,
            ordering="blocked",
            boson_encoding="binary",
            blocks=(),
        ),
        hamiltonian=object(),
        sector=SectorSelection(
            label="test_sector",
            comparison_space_label="test_space",
            constraints=(),
            num_particles=(1, 1),
        ),
        reference_state=ReferenceStateSpec(
            kind="test",
            source_label="test",
            state_kind="statevector",
            build_state=lambda: np.asarray([1.0, 0.0, 0.0, 0.0]),
        ),
        exact_target=ExactTargetSpec(
            kind="test",
            comparison_space_label="test_space",
            resolve_energy=lambda **_kwargs: -1.0,
            exact_state_policy="test",
            build_fallback_anchor_state=lambda: np.asarray(
                [1.0, 0.0, 0.0, 0.0]
            ),
            fallback_policy="test",
        ),
        default_controller_profile="test",
        default_continuation_mode="test",
        admissible_pool_keys=("full_meta",),
        default_pool_key="full_meta",
        default_pool_resolution_scope="test",
        default_sector_label="test_sector",
        default_reference_label="test",
        exact_target_label="test",
        exact_comparison_space_label="test_space",
        default_num_particles=(1, 1),
        capabilities=HamiltonianFamilyCapabilities(),
    )


def _inventory(
    representation: str,
    label: str,
    *,
    schema: str,
) -> CandidateInventory:
    labels = (label,)
    record = CandidateRecord(
        label=label,
        term=AnsatzTerm(
            label=label,
            polynomial=PauliPolynomial(
                "JW", [PauliTerm(2, ps="xe", pc=1.0)]
            ),
        ),
        representation_id=representation,
        generator_identity=f"test:{label}",
        parent_identities=(),
        family_id="test",
        stage_family="test",
        construction="test",
        execution_mode="termwise_product",
        serialized_terms_exyz=(
            {
                "pauli_exyz": "xe",
                "coeff_real": 1.0,
                "coeff_imag": 0.0,
            },
        ),
    )
    receipt = PoolInventoryReceipt(
        schema=schema,
        candidate_representation=representation,
        ordered_labels=labels,
        ordered_labels_sha256=canonical_sha256(list(labels)),
        ordered_pool_sha256=canonical_sha256(
            [{"label": label, "representation": representation}]
        ),
        count=1,
    )
    return CandidateInventory(
        candidates=(record,),
        receipt=receipt,
        metadata={"test_fixture": True},
    )


def _patch_macro_inventories(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CandidateInventory, CandidateInventory]:
    parent = _inventory(
        CANDIDATE_REPRESENTATION_MACRO,
        "parent",
        schema="test_parent_v1",
    )
    executable = _inventory(
        CANDIDATE_REPRESENTATION_MACRO,
        "macro",
        schema="test_macro_v1",
    )
    monkeypatch.setattr(
        MacroCandidateAdapter,
        "parent_inventory",
        lambda _self, _problem: parent,
    )
    monkeypatch.setattr(
        MacroCandidateAdapter,
        "executable_pool",
        lambda _self, _problem: executable,
    )
    return parent, executable


def _patch_single_inventories(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CandidateInventory, CandidateInventory]:
    parent = _inventory(
        CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        "parent",
        schema="test_parent_v1",
    )
    children = _inventory(
        CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        "child",
        schema="test_guarded_children_v1",
    )
    monkeypatch.setattr(
        SinglePauliWordCandidateAdapter,
        "parent_inventory",
        lambda _self, _problem: parent,
    )
    monkeypatch.setattr(
        SinglePauliWordCandidateAdapter,
        "executable_pool",
        lambda *_args, **_kwargs: pytest.fail(
            "Append must not use the staged singleton child factory."
        ),
    )
    monkeypatch.setattr(
        SinglePauliWordCandidateAdapter,
        "global_executable_pool",
        lambda _self, _problem: children,
    )
    return parent, children


def _execution_payload(protocol: Any) -> dict[str, Any]:
    label = str(protocol.executable_pool.ordered_labels[0])
    generator_identity = f"test:{label}"
    components = {
        "N_H_outer": 1,
        "N_H_refit": 1,
        "N_grad": 1,
        "N_metric": 0,
    }
    occurrence_summary = {**components, "S_alg": 3}
    ledger = {
        "schema": "estimator_call_ledger_v1",
        "ledger_fingerprint": canonical_sha256(
            {
                "protocol_sha256": protocol.sha256,
                "fixture": "append_summary",
            }
        ),
        "entries": [],
        "occurrences": [
            {"sequence": sequence} for sequence in range(1, 4)
        ],
        "summary": {},
        "occurrence_summary": occurrence_summary,
    }
    terminal_resources = {
        "compiled_circuit_stats_status": "ok",
        "compiled_resource_source_kind": (
            "paper_i_append_adapt_terminal_ansatz_v1"
        ),
        "compiled_resource_qiskit_validated": True,
        "compile_convention": "table_i_basis_gate_transpile_v1",
    }
    closed_prefix = {
        "schema": "estimator_call_ledger_occurrence_prefix_summary_v1",
        "component_contract": [
            "N_H_outer",
            "N_H_refit",
            "N_grad",
            "N_metric",
        ],
        "occurrence_sequence_end_inclusive": 3,
        "cumulative_raw_occurrences": {
            "components": components,
            "total": 3,
        },
        "cumulative_executed_queries": {
            "components": components,
            "S_alg": 3,
            "unit": "executed_logical_scalar_estimator_invocation",
        },
        "cumulative_unique_primitives": {
            "components": components,
            "S_unique": 3,
        },
        "unique_primitive_count": 3,
        "primitive_set_sha256": canonical_sha256(
            {"fixture": "append_summary_primitives"}
        ),
    }
    accepted_refit = {
        "schema": "test_append_native_accepted_refit_v1",
        "final_energy": -1.0,
    }
    active_prefix_checkpoint = build_signed_append_prefix_checkpoint(
        protocol=protocol,
        controller_round=1,
        accepted_operator_labels=(label,),
        accepted_generator_identities=(generator_identity,),
        logical_parameters=(0.0,),
        runtime_parameters=(0.0,),
        projective_state_fingerprint=(
            "projective_state_v1:test_append_summary"
        ),
        accepted_energy=-1.0,
        accepted_refit=accepted_refit,
        estimator_prefix=closed_prefix,
    )
    payload = {
        "schema": append.APPEND_EXECUTION_SCHEMA,
        "algorithm_id": protocol.algorithm_id,
        "protocol_sha256": protocol.sha256,
        "selector_identity": APPEND_CONVENTIONAL_SELECTOR_ID,
        "selector_scope": APPEND_CONVENTIONAL_SELECTOR_SCOPE,
        "selector_source_id": append.APPEND_SELECTOR_SOURCE_ID,
        "candidate_representation": protocol.candidate_representation,
        "selection_with_replacement": True,
        "append_position_only": True,
        "ra_staged_funnel_invoked": False,
        "accepted_operator_labels": [label],
        "accepted_generator_identities": [generator_identity],
        "controller_rounds_completed": 1,
        "stop_reason": "maximum_controller_rounds",
        "final_energy": -1.0,
        "logical_theta": [0.0],
        "runtime_theta": [0.0],
        "history": [
            {
                "controller_round": 1,
                "selected_label": label,
                "selected_generator_identity": generator_identity,
                "insertion_position": 0,
                "energy_before": -0.5,
                "energy_after": -1.0,
                "selected_abs_commutator_gradient": 1.0,
                "accepted_refit": accepted_refit,
                "active_prefix_checkpoint": active_prefix_checkpoint,
            }
        ],
        "estimator_accounting": {
            "schema": "paper_i_append_estimator_accounting_v2",
            "convention": protocol.estimator_accounting_convention,
            "components": components,
            **components,
            "S_alg": 3,
            "closed_occurrence_reconciliation": True,
            "occurrence_summary": occurrence_summary,
            "closed_occurrence_prefix": closed_prefix,
        },
        "estimator_call_ledger": ledger,
        "compile_identity": dict(protocol.compile_identity),
        "compiled_resources": terminal_resources,
        "resource_observation": {
            "requested_resource_rounds": [],
            "materialized_resource_rounds": [],
            "unmaterialized_resource_rounds": [],
        },
        "compiled_resources_by_round": [],
        "accepted_refit_scope": protocol.accepted_refit_scope,
        "accepted_refit_coordinate_chart": (
            protocol.accepted_refit_coordinate_chart
        ),
    }
    payload["controller_replay_evidence"] = (
        build_append_controller_replay_evidence(
            protocol=protocol,
            history=payload["history"],
            estimator_ledger=payload["estimator_call_ledger"],
            estimator_accounting=payload["estimator_accounting"],
        )
    )
    payload["numerical_physical_integrity"] = (
        build_append_numerical_physical_integrity(
            problem=_problem(),
            final_state=_problem().reference_state.build_state(),
            history=payload["history"],
            logical_parameters=payload["logical_theta"],
            runtime_parameters=payload["runtime_theta"],
            final_energy=payload["final_energy"],
        ).to_dict()
    )
    return payload


def _zero_acceptance_execution_payload(protocol: Any) -> dict[str, Any]:
    payload = _execution_payload(protocol)
    components = {
        "N_H_outer": 1,
        "N_H_refit": 0,
        "N_grad": 1,
        "N_metric": 0,
    }
    occurrence_summary = {**components, "S_alg": 2}
    closed_prefix = {
        "schema": "estimator_call_ledger_occurrence_prefix_summary_v1",
        "component_contract": [
            "N_H_outer",
            "N_H_refit",
            "N_grad",
            "N_metric",
        ],
        "occurrence_sequence_end_inclusive": 2,
        "cumulative_raw_occurrences": {
            "components": components,
            "total": 2,
        },
        "cumulative_executed_queries": {
            "components": components,
            "S_alg": 2,
            "unit": "executed_logical_scalar_estimator_invocation",
        },
        "cumulative_unique_primitives": {
            "components": components,
            "S_unique": 2,
        },
        "unique_primitive_count": 2,
        "primitive_set_sha256": canonical_sha256(
            {"fixture": "append_zero_acceptance_primitives"}
        ),
    }
    payload.update(
        accepted_operator_labels=[],
        accepted_generator_identities=[],
        controller_rounds_completed=0,
        stop_reason="initial_gradient_stop",
        final_energy=-0.5,
        logical_theta=[],
        runtime_theta=[],
        history=[],
        estimator_accounting={
            "schema": "paper_i_append_estimator_accounting_v2",
            "convention": protocol.estimator_accounting_convention,
            "components": components,
            **components,
            "S_alg": 2,
            "closed_occurrence_reconciliation": True,
            "occurrence_summary": occurrence_summary,
            "closed_occurrence_prefix": closed_prefix,
        },
        estimator_call_ledger={
            "schema": "estimator_call_ledger_v1",
            "ledger_fingerprint": canonical_sha256(
                {
                    "protocol_sha256": protocol.sha256,
                    "fixture": "append_zero_acceptance_summary",
                }
            ),
            "entries": [],
            "occurrences": [
                {"sequence": sequence} for sequence in range(1, 3)
            ],
            "summary": {},
            "occurrence_summary": occurrence_summary,
        },
    )
    payload["controller_replay_evidence"] = (
        build_append_controller_replay_evidence(
            protocol=protocol,
            history=payload["history"],
            estimator_ledger=payload["estimator_call_ledger"],
            estimator_accounting=payload["estimator_accounting"],
        )
    )
    payload["numerical_physical_integrity"] = (
        build_append_numerical_physical_integrity(
            problem=_problem(),
            final_state=_problem().reference_state.build_state(),
            history=payload["history"],
            logical_parameters=payload["logical_theta"],
            runtime_parameters=payload["runtime_theta"],
            final_energy=payload["final_energy"],
        ).to_dict()
    )
    return payload


def _assert_canonical_append_summary(
    result: Any,
    *,
    candidate_representation: str,
) -> None:
    summary = result.paper_i_summary
    assert summary.schema == PAPER_I_APPEND_RUN_SUMMARY_SCHEMA
    assert summary.protocol_sha256 == result.protocol.sha256
    assert summary.candidate_representation == candidate_representation
    assert summary.selector_identity == APPEND_CONVENTIONAL_SELECTOR_ID
    assert summary.selector_scope == APPEND_CONVENTIONAL_SELECTOR_SCOPE
    assert summary.controller_rounds_completed == 1
    assert summary.accepted_operator_labels == tuple(
        result.result_payload["accepted_operator_labels"]
    )
    assert summary.accepted_generator_identities == tuple(
        result.result_payload["accepted_generator_identities"]
    )
    assert summary.estimator_accounting.S_alg == 3
    assert summary.estimator_accounting.closed_occurrence_reconciliation
    assert summary.resources.terminal_observation_status == "ok"
    assert summary.resources.accepted_prefix_observation_status == (
        "not_requested"
    )
    assert summary.additional_estimator_acquisitions == 0
    assert summary.additional_controller_rounds == 0
    assert summary.to_json().encode("utf-8") == canonical_json_bytes(
        summary
    )
    assert result.to_dict()["paper_i_summary"] == summary.to_dict()
    assert result.scientific_receipts[
        "paper_i_append_run_summary"
    ] == summary.to_dict()
    assert result.scientific_receipts[
        "paper_i_append_run_summary_sha256"
    ] == canonical_sha256(summary)
    integrity = result.numerical_physical_integrity
    assert integrity.reporting_only is True
    assert integrity.controller_decision_influence is False
    assert integrity.finite_values_passed is True
    assert integrity.sector_leak_flag is False
    assert integrity.boson_truncation_leak_flag is False
    assert integrity.accepted_energy_integrity_passed is True
    assert integrity.integrity_passed is True
    assert result.result_payload[
        "numerical_physical_integrity"
    ] == integrity.to_dict()
    assert result.scientific_receipts[
        "numerical_physical_integrity"
    ] == integrity.to_dict()
    assert result.scientific_receipts[
        "numerical_physical_integrity_sha256"
    ] == canonical_sha256(integrity)


def _patch_one_candidate_numerical_inventory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    multi_pauli: bool = False,
) -> ResolvedProblemContext:
    problem = _problem()
    problem = ResolvedProblemContext(
        **{
            **problem.__dict__,
            "hamiltonian": PauliPolynomial(
                "JW", [PauliTerm(2, ps="xe", pc=1.0)]
            ),
        }
    )
    polynomial_terms = [PauliTerm(2, ps="ye", pc=1.0)]
    serialized_terms = [
        {
            "pauli_exyz": "ye",
            "coeff_real": 1.0,
            "coeff_imag": 0.0,
        }
    ]
    if multi_pauli:
        polynomial_terms.append(PauliTerm(2, ps="ey", pc=1.0))
        serialized_terms.append(
            {
                "pauli_exyz": "ey",
                "coeff_real": 1.0,
                "coeff_imag": 0.0,
            }
        )
    term = AnsatzTerm(
        label="y_rotation",
        polynomial=PauliPolynomial("JW", polynomial_terms),
    )
    record = CandidateRecord(
        label="y_rotation",
        term=term,
        representation_id=CANDIDATE_REPRESENTATION_MACRO,
        generator_identity="test:y_rotation",
        parent_identities=(),
        family_id="test",
        stage_family="test",
        construction="test",
        execution_mode="termwise_product",
        serialized_terms_exyz=tuple(serialized_terms),
    )
    inventory = CandidateInventory(
        candidates=(record,),
        receipt=_receipt(
            schema="test_numerical_macro_v1",
            representation_id=CANDIDATE_REPRESENTATION_MACRO,
            candidates=(record,),
        ),
        metadata={"test_fixture": True},
    )
    monkeypatch.setattr(
        MacroCandidateAdapter,
        "parent_inventory",
        lambda _self, _problem: inventory,
    )
    monkeypatch.setattr(
        MacroCandidateAdapter,
        "executable_pool",
        lambda _self, _problem: inventory,
    )
    return problem


def test_public_signature_is_problem_request_only() -> None:
    signature = inspect.signature(append.run_append_adapt)
    assert tuple(signature.parameters) == ("problem", "request")
    assert signature.parameters["request"].default is None
    resolver = inspect.signature(append.build_resolved_append_protocol)
    assert tuple(resolver.parameters) == (
        "problem",
        "request",
        "materialization_authority",
    )


def test_macro_facade_uses_common_executable_pool_and_not_ra_funnel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, executable = _patch_macro_inventories(monkeypatch)
    observed: dict[str, Any] = {}

    def _execute(problem, protocol, inventory):
        observed.update(
            problem=problem, protocol=protocol, inventory=inventory
        )
        return _execution_payload(protocol)

    monkeypatch.setattr(append, "_execute_conventional_append", _execute)
    # If the facade accidentally delegates into the RA engine this makes the
    # separation failure explicit.
    from pipelines.static_adapt.ra_adapt import engine

    monkeypatch.setattr(
        engine,
        "run_ra_adapt",
        lambda *_args, **_kwargs: pytest.fail(
            "Append invoked the RA P1/P2/P3 funnel."
        ),
    )
    result = append.run_append_adapt(
        _problem(),
        AppendAdaptRequest(
            adapter=MacroCandidateAdapter(),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
        ),
    )

    assert observed["inventory"] is executable
    assert result.parent_inventory == parent.receipt
    assert result.executable_pool == executable.receipt
    assert result.selector_identity == APPEND_CONVENTIONAL_SELECTOR_ID
    assert result.protocol.selector_identity == APPEND_CONVENTIONAL_SELECTOR_ID
    assert (
        result.protocol.selector_scope
        == APPEND_CONVENTIONAL_SELECTOR_SCOPE
    )
    assert result.protocol.accepted_refit_coordinate_chart == (
        NATIVE_REFIT_CHART
    )
    assert result.protocol.accepted_refit_base_chart_policy == (
        LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART
    )
    assert result.result_payload["selector_scope"] == (
        APPEND_CONVENTIONAL_SELECTOR_SCOPE
    )
    assert result.scientific_receipts["selector_scope"] == (
        APPEND_CONVENTIONAL_SELECTOR_SCOPE
    )
    assert result.scientific_receipts[
        "accepted_refit_coordinate_chart"
    ] == NATIVE_REFIT_CHART
    assert result.scientific_receipts["ra_staged_funnel_invoked"] is False
    assert result.scientific_receipts["executable_pool_sha256"] == (
        executable.receipt.ordered_pool_sha256
    )
    _assert_canonical_append_summary(
        result,
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
    )


def test_singleton_facade_builds_global_guarded_pool_before_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, children = _patch_single_inventories(monkeypatch)
    observed: dict[str, Any] = {}

    def _execute(_problem, protocol, inventory):
        observed["inventory"] = inventory
        payload = _execution_payload(protocol)
        payload["global_pool_constructed_before_gradient_selection"] = True
        return payload

    monkeypatch.setattr(append, "_execute_conventional_append", _execute)
    result = append.run_append_adapt(
        _problem(),
        AppendAdaptRequest(
            adapter=SinglePauliWordCandidateAdapter(),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
        ),
    )

    assert observed["inventory"] is children
    assert result.parent_inventory == parent.receipt
    assert result.executable_pool == children.receipt
    assert result.protocol.candidate_representation == (
        CANDIDATE_REPRESENTATION_SINGLE_PAULI
    )
    assert result.protocol.selector_scope == (
        APPEND_CONVENTIONAL_SELECTOR_SCOPE
    )
    assert result.result_payload[
        "global_pool_constructed_before_gradient_selection"
    ] is True
    _assert_canonical_append_summary(
        result,
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    )


def test_summary_supports_valid_zero_acceptance_terminal_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_macro_inventories(monkeypatch)
    execution_calls = 0

    def _execute(_problem, protocol, _inventory):
        nonlocal execution_calls
        execution_calls += 1
        return _zero_acceptance_execution_payload(protocol)

    monkeypatch.setattr(append, "_execute_conventional_append", _execute)
    result = append.run_append_adapt(
        _problem(),
        AppendAdaptRequest(
            adapter=MacroCandidateAdapter(),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
        ),
    )

    summary = result.paper_i_summary
    assert execution_calls == 1
    assert summary.controller_rounds_completed == 0
    assert summary.available_controller_rounds == 0
    assert summary.accepted_operator_labels == ()
    assert summary.accepted_generator_identities == ()
    assert summary.accepted_history == ()
    assert summary.final_energy == pytest.approx(-0.5)
    assert summary.stop_reason == "initial_gradient_stop"
    assert summary.estimator_accounting.S_alg == 2
    assert summary.estimator_accounting.N_H_refit == 0
    assert summary.estimator_accounting.closed_occurrence_reconciliation
    assert summary.additional_estimator_acquisitions == 0
    assert summary.additional_controller_rounds == 0
    assert result.scientific_receipts[
        "paper_i_append_run_summary"
    ] == summary.to_dict()
    integrity = result.numerical_physical_integrity
    assert integrity.accepted_energy_transitions == ()
    assert integrity.accepted_energy_integrity_passed is True
    assert integrity.integrity_passed is True
    replay = validate_controller_replay_evidence(
        result.scientific_receipts["controller_replay_evidence"]
    )
    assert replay["signed_controller_round_prefixes"] == []
    assert replay["resume_sidecar_closure"][
        "zero_acceptance_terminal"
    ] is True


def test_bundle_only_policy_inputs_have_no_direct_builder_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_macro_inventories(monkeypatch)
    with pytest.raises(TypeError, match="unexpected keyword"):
        append.build_resolved_append_protocol(
            _problem(),
            AppendAdaptRequest(
                adapter=MacroCandidateAdapter(),
                execution=SRExecutionPolicy(
                    stop=SRStopPolicy(maximum_controller_rounds=3)
                ),
            ),
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
        )
    with pytest.raises(TypeError, match="minted only"):
        BundleProtocolMaterializationAuthority()


def test_raw_append_study_protocol_requires_validated_bundle_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_macro_inventories(monkeypatch)
    problem = _problem()
    request = AppendAdaptRequest(
        adapter=MacroCandidateAdapter(),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=1)
        ),
    )
    cell = bundle_module.BundleCellSpec(
        cell_id="raw_append_study_protocol",
        stage="validation",
        regime_id="fixture",
        nph=3,
        route_id=bundle_module.ROUTE_APPEND_MACRO,
        algorithm_id=append.APPEND_ADAPT_ALGORITHM_ID,
        selector_family="append_adapt",
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
        horizon=1,
        source_lock_id="fixture_lock",
    )
    refs = {
        "source_locks_manifest_sha256": "1" * 64,
        "implementation_source_inventory_sha256": "2" * 64,
        "cell_source_lock_id": "fixture_lock",
        "cell_source_lock_sha256": "3" * 64,
        "visible_provenance_sha256": "4" * 64,
        "provenance_tracker_sha256": "5" * 64,
        "ed_cutoff_reference_sha256": "6" * 64,
        "resolver_script_sha256": "7" * 64,
    }
    authority = (
        bundle_module._bundle_protocol_materialization_authority(
            cell=cell,
            bundle_id="fixture_stationary_late_v1",
            bundle_manifest_sha256="8" * 64,
            source_locks_sha256="1" * 64,
            source_lock_refs=refs,
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
        )
    )
    protocol = append.build_resolved_append_protocol(
        problem,
        request,
        materialization_authority=authority,
    )
    raw_protocol = resolved_ra_adapt_protocol_from_mapping(
        protocol.to_dict()
    )
    monkeypatch.setattr(
        append,
        "_append_inventories",
        lambda *_args, **_kwargs: pytest.fail(
            "Pool construction preceded bundle authentication."
        ),
    )
    with pytest.raises(ValueError, match="load_validated_bundle_protocol"):
        append.run_append_adapt(problem, raw_protocol)


def test_resolved_protocol_fails_closed_on_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_macro_inventories(monkeypatch)
    problem = _problem()
    protocol = append.build_resolved_append_protocol(
        problem,
        AppendAdaptRequest(
            adapter=MacroCandidateAdapter(),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
        ),
    )
    original = append._source_lock_receipts(problem)
    monkeypatch.setattr(
        append,
        "_source_lock_receipts",
        lambda _problem: {
            **original,
            "selector_module_sha256": "f" * 64,
        },
    )

    with pytest.raises(ValueError, match="selector_module_sha256.*drifted"):
        append.run_append_adapt(problem, protocol)


def test_resolved_protocol_rejects_nested_pool_receipt_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_macro_inventories(monkeypatch)
    problem = _problem()
    protocol = append.build_resolved_append_protocol(
        problem,
        AppendAdaptRequest(
            adapter=MacroCandidateAdapter(),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
        ),
    )
    drifted_pool = replace(
        protocol.parent_inventory,
        candidate_representation=(
            CANDIDATE_REPRESENTATION_SINGLE_PAULI
        ),
        sha256=None,
    )
    digest_payload = protocol.digest_payload()
    digest_payload["parent_inventory"] = drifted_pool.to_dict()
    drifted_protocol = replace(
        protocol,
        parent_inventory=drifted_pool,
        sha256=canonical_sha256(digest_payload),
    )
    with pytest.raises(ValueError, match="parent inventory drifted"):
        append.run_append_adapt(problem, drifted_protocol)


@pytest.mark.parametrize(
    ("family", "num_sites"),
    (("hubbard", 2), ("hh", 3)),
)
def test_facade_retains_paper_i_hh_l2_lock(
    family: str,
    num_sites: int,
) -> None:
    with pytest.raises(ValueError, match="Hubbard--Holstein L=2"):
        append.run_append_adapt(
            _problem(family=family, num_sites=num_sites)
        )


def test_conventional_selector_is_largest_absolute_gradient_with_label_tie() -> None:
    winner = append._select_largest_absolute_commutator_gradient(
        (
            {"label": "z", "abs_gradient": 2.0},
            {"label": "b", "abs_gradient": 3.0},
            {"label": "a", "abs_gradient": 3.0},
        )
    )
    assert winner is not None
    assert winner["label"] == "a"
    assert append._select_largest_absolute_commutator_gradient(()) is None


def test_executor_attestation_rejects_ra_funnel_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_macro_inventories(monkeypatch)

    def _bad_execute(_problem, protocol, _inventory):
        payload = _execution_payload(protocol)
        payload["ra_staged_funnel_invoked"] = True
        return payload

    monkeypatch.setattr(
        append, "_execute_conventional_append", _bad_execute
    )
    with pytest.raises(RuntimeError, match="separation from the RA funnel"):
        append.run_append_adapt(
            _problem(),
            AppendAdaptRequest(
                adapter=MacroCandidateAdapter(),
                execution=SRExecutionPolicy(
                    stop=SRStopPolicy(maximum_controller_rounds=1)
                ),
            ),
        )


def test_executor_attestation_rejects_foreign_selector_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_macro_inventories(monkeypatch)

    def _bad_execute(_problem, protocol, _inventory):
        payload = _execution_payload(protocol)
        payload["selector_scope"] = "foreign_selector_scope"
        return payload

    monkeypatch.setattr(
        append, "_execute_conventional_append", _bad_execute
    )
    with pytest.raises(RuntimeError, match="selector scope"):
        append.run_append_adapt(
            _problem(),
            AppendAdaptRequest(
                adapter=MacroCandidateAdapter(),
                execution=SRExecutionPolicy(
                    stop=SRStopPolicy(maximum_controller_rounds=1)
                ),
            ),
        )


def test_one_round_numerical_smoke_uses_native_logical_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _patch_one_candidate_numerical_inventory(monkeypatch)

    result = append.run_append_adapt(
        problem,
        AppendAdaptRequest(
            adapter=MacroCandidateAdapter(),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
        ),
    )

    payload = result.result_payload
    assert payload["accepted_operator_labels"] == ["y_rotation"]
    assert payload["final_energy"] == pytest.approx(-1.0, abs=1.0e-10)
    integrity = result.numerical_physical_integrity
    assert integrity.finite_values_passed is True
    assert integrity.sector_leak_flag is False
    assert integrity.boson_truncation_leak_flag is False
    assert integrity.accepted_energy_integrity_passed is True
    assert integrity.integrity_passed is True
    assert payload["numerical_physical_integrity"] == (
        integrity.to_dict()
    )
    assert payload["history"][0]["candidate_geometry"][
        "coordinate_chart"
    ] == "exact_ordered_insertion_zero_angle_v1"
    selected_lineage = payload["history"][0][
        "selected_candidate_lineage"
    ]
    assert selected_lineage["candidate_label"] == "y_rotation"
    assert selected_lineage["generator_identity"] == "test:y_rotation"
    assert selected_lineage["parent_identities"] == []
    assert selected_lineage["insertion_position"] == 0
    assert len(selected_lineage["candidate_manifest_sha256"]) == 64
    accepted = payload["history"][0]["accepted_refit"]
    accepted_digest = accepted.pop("sha256")
    assert accepted_digest == canonical_sha256(accepted)
    accepted["sha256"] = accepted_digest
    assert accepted["schema"] == (
        append.APPEND_NATIVE_REFIT_RECEIPT_SCHEMA
    )
    assert accepted["coordinate_chart"] == NATIVE_REFIT_CHART
    assert accepted["base_chart_policy"] == (
        LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART
    )
    assert accepted["base_chart_applied"] is None
    assert accepted["optimizer_coordinate_mode"] == "logical_shared"
    assert accepted["logical_parameter_count"] == 1
    assert accepted["optimizer_parameter_count"] == 1
    assert accepted["runtime_parameter_count"] == 1
    assert accepted["admitted_coordinate_initialized_to_zero"] is True
    assert accepted["chart_fixed_within_powell_invocation"] is True
    assert accepted["supported_fs_chart_constructed"] is False
    assert accepted["whitening_performed"] is False
    assert accepted["metric_backend_evaluation_performed"] is False
    assert accepted["chart_origin_hamiltonian_acquisition_count"] == 0
    assert accepted["chart_origin_gradient_acquisition_count"] == 0
    assert accepted["chart_origin_metric_acquisition_count"] == 0
    assert "accepted_refit_fixed_chart_receipt" not in accepted
    accounting = payload["estimator_accounting"]
    assert accounting["S_alg"] == sum(
        int(accounting[key])
        for key in ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
    )
    assert accounting["occurrence_summary"]["S_alg"] == accounting["S_alg"]
    assert (
        accounting["closed_occurrence_prefix"][
            "cumulative_executed_queries"
        ]["S_alg"]
        == accounting["S_alg"]
    )
    instrumentation = accounting[
        "executed_occurrence_instrumentation"
    ]
    assert instrumentation["closed_against_estimator_ledger"] is True
    assert instrumentation["N_H_outer_and_refit"] == (
        accounting["N_H_outer"] + accounting["N_H_refit"]
    )
    assert instrumentation["N_grad"] == accounting["N_grad"]
    assert instrumentation["N_metric"] == accounting["N_metric"]
    assert accounting["N_metric"] == 0
    assert accounting["N_grad"] == sum(
        int(row["candidate_count_scored"]) for row in payload["history"]
    )
    ledger = payload["estimator_call_ledger"]
    assert ledger["schema"] == "estimator_call_ledger_v1"
    ledger_before_summary_projection = canonical_json_bytes(ledger)
    rebuilt_summary = summarize_paper_i_append_run(
        protocol=result.protocol,
        selector_identity=result.selector_identity,
        result_payload=payload,
    )
    assert rebuilt_summary == result.paper_i_summary
    assert canonical_json_bytes(ledger) == ledger_before_summary_projection
    assert rebuilt_summary.estimator_accounting.S_alg == accounting["S_alg"]
    assert rebuilt_summary.estimator_accounting.N_H_outer == (
        accounting["N_H_outer"]
    )
    assert rebuilt_summary.estimator_accounting.N_H_refit == (
        accounting["N_H_refit"]
    )
    assert rebuilt_summary.estimator_accounting.N_grad == (
        accounting["N_grad"]
    )
    assert rebuilt_summary.estimator_accounting.N_metric == (
        accounting["N_metric"]
    )
    assert rebuilt_summary.estimator_accounting.ledger_occurrence_count == (
        accounting["S_alg"]
    )
    assert rebuilt_summary.estimator_accounting.ledger_sha256 == (
        canonical_sha256(ledger)
    )
    assert rebuilt_summary.additional_estimator_acquisitions == 0
    assert rebuilt_summary.additional_controller_rounds == 0
    assert not any(
        occurrence["consumer_scope"].endswith("accepted_refit_origin")
        for occurrence in ledger["occurrences"]
    )
    refit_occurrences = [
        occurrence
        for occurrence in ledger["occurrences"]
        if occurrence["component"] == "N_H_refit"
    ]
    assert refit_occurrences
    assert all(
        occurrence["consumer_scope"].endswith("accepted_refit_powell")
        for occurrence in refit_occurrences
    )
    resources = payload["compiled_resources"]
    assert resources["compiled_resource_source_kind"] == (
        "paper_i_append_adapt_terminal_ansatz_v1"
    )
    assert resources["compile_convention"] == (
        "table_i_basis_gate_transpile_v1"
    )
    assert "depth_proxy" not in resources
    assert payload["ra_staged_funnel_invoked"] is False
    assert payload["selector_scope"] == APPEND_CONVENTIONAL_SELECTOR_SCOPE
    assert payload["accepted_refit_coordinate_chart"] == (
        NATIVE_REFIT_CHART
    )
    assert result.scientific_receipts["selector_scope"] == (
        APPEND_CONVENTIONAL_SELECTOR_SCOPE
    )
    inventory_lineage = result.scientific_receipts[
        "candidate_inventory_lineage"
    ]
    assert inventory_lineage["count"] == 1
    assert inventory_lineage["ordered_rows"][0] == {
        "label": "y_rotation",
        "representation_id": CANDIDATE_REPRESENTATION_MACRO,
        "generator_identity": "test:y_rotation",
        "parent_identities": [],
    }


def test_native_refit_uses_one_logical_coordinate_for_multi_pauli_macro(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _patch_one_candidate_numerical_inventory(
        monkeypatch,
        multi_pauli=True,
    )

    result = append.run_append_adapt(
        problem,
        AppendAdaptRequest(
            adapter=MacroCandidateAdapter(),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
        ),
    )

    payload = result.result_payload
    accepted = payload["history"][0]["accepted_refit"]
    assert accepted["optimizer_coordinate_mode"] == "logical_shared"
    assert accepted["logical_parameter_count"] == 1
    assert accepted["optimizer_parameter_count"] == 1
    assert accepted["runtime_parameter_count"] == 2
    assert len(accepted["origin_logical_theta"]) == 1
    assert len(accepted["origin_runtime_theta"]) == 2
    assert len(payload["logical_theta"]) == 1
    assert len(payload["runtime_theta"]) == 2
    assert payload["estimator_accounting"]["N_metric"] == 0
    assert payload["estimator_accounting"]["N_grad"] == 1


def test_observation_honors_checkpoint_cadence_tail_and_resource_rounds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    problem = _patch_one_candidate_numerical_inventory(monkeypatch)
    compile_calls: list[str] = []

    def _fake_compile(**kwargs: Any) -> dict[str, Any]:
        source_kind = str(kwargs["source_kind"])
        compile_calls.append(source_kind)
        return {
            "compiled_circuit_stats_status": "ok",
            "compiled_resource_source_kind": source_kind,
            "compiled_resource_qiskit_validated": True,
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "qiskit_version": "test",
        }

    monkeypatch.setattr(
        append, "_compile_append_resources", _fake_compile
    )
    checkpoint_path = tmp_path / "append_checkpoint.json"
    result = append.run_append_adapt(
        problem,
        AppendAdaptRequest(
            adapter=MacroCandidateAdapter(),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=1)
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=checkpoint_path,
                    every_controller_rounds=1,
                    keep_history_tail=0,
                ),
                resource_rounds=(1,),
            ),
        ),
    )

    assert compile_calls == [
        "paper_i_append_adapt_accepted_prefix_v1",
        "paper_i_append_adapt_terminal_ansatz_v1",
    ]
    observation = result.result_payload["resource_observation"]
    assert observation == {
        "requested_resource_rounds": [1],
        "materialized_resource_rounds": [1],
        "unmaterialized_resource_rounds": [],
    }
    prefix = result.result_payload["compiled_resources_by_round"][0]
    assert prefix["controller_round"] == 1
    assert prefix["accepted_prefix_length"] == 1
    summary_resources = result.paper_i_summary.resources
    assert summary_resources.terminal_observation_status == "ok"
    assert summary_resources.accepted_prefix_observation_status == "complete"
    assert summary_resources.requested_controller_rounds == (1,)
    assert summary_resources.materialized_controller_rounds == (1,)
    assert summary_resources.unmaterialized_controller_rounds == ()
    assert len(summary_resources.compiled_resources_by_round) == 1
    assert (
        summary_resources.compiled_resources_by_round[0].observation_status
        == "ok"
    )
    assert compile_calls == [
        "paper_i_append_adapt_accepted_prefix_v1",
        "paper_i_append_adapt_terminal_ansatz_v1",
    ]

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["schema"] == "paper_i_append_adapt_checkpoint_v1"
    assert checkpoint["status"] == "completed"
    assert checkpoint["history_total_count"] == 1
    assert checkpoint["history_tail"] == []
    assert checkpoint["compiled_resources_by_round"][0][
        "controller_round"
    ] == 1
    digest = checkpoint.pop("sha256")
    assert digest == canonical_sha256(checkpoint)


def test_fixed_horizon_macro_sequence_matches_frozen_comparator_on_common_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pipelines.exact_bench import (
        generic_static_adapt_variants as frozen,
    )

    case_id = (
        frozen.TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS[
            "weak-weak"
        ]
    )
    spec = frozen._spec_by_case_id(
        "hh",
        case_id,
        frozen.STATIC_FULL_META_APPEND_ADAPT_VQE,
    )
    problem = frozen._resolve_context_from_spec(spec)
    adapter = MacroCandidateAdapter()
    inventory = adapter.executable_pool(problem)
    assert inventory.receipt.count == 102

    frozen_candidates = []
    for record in inventory.candidates:
        labels, support = frozen._polynomial_labels_and_support(
            record.term.polynomial
        )
        frozen_candidates.append(
            frozen._PoolCandidate(
                label=record.label,
                polynomial=record.term.polynomial,
                support=tuple(support),
                pauli_labels_exyz=tuple(labels),
                construction="frozen_common_macro_pool_v1",
                execution_mode=record.execution_mode,
                generator_metadata=dict(record.generator_metadata),
            )
        )
    frozen_pool = frozen._FullMetaCandidatePoolResult(
        candidates=tuple(frozen_candidates),
        selected_logical_filter_meta=None,
        pool_key="full_meta",
    )
    monkeypatch.setattr(
        frozen,
        "_build_full_meta_candidate_pool_with_meta",
        lambda *_args, **_kwargs: frozen_pool,
    )
    frozen_result = frozen._run_impl(
        family="hh",
        case_id=case_id,
        algorithm_id=frozen.STATIC_FULL_META_APPEND_ADAPT_VQE,
        output_dir=tmp_path / "frozen",
        max_adapt_iterations=2,
        optimizer_maxiter=200,
        gradient_threshold=0.0,
        seed=7,
        generic_adapt_stop_policy="fixed_horizon_no_target_v1",
        powell_maxiter_cap_policy="strict_failure_v1",
        hh_adaptive_pool_profile="full_meta_unfiltered",
    )
    canonical_result = append.run_append_adapt(
        problem,
        AppendAdaptRequest(
            adapter=adapter,
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=2)
            ),
        ),
    )

    assert frozen_result["status"] == "completed"
    assert frozen_result["result"]["pool_term_count"] == 102
    assert canonical_result.executable_pool == inventory.receipt
    assert canonical_result.result_payload[
        "accepted_operator_labels"
    ] == frozen_result["result"]["selected_operators"]
    assert canonical_result.result_payload["final_energy"] == pytest.approx(
        frozen_result["result"]["energy"],
        rel=1.0e-10,
        abs=1.0e-10,
    )
    np.testing.assert_allclose(
        canonical_result.result_payload["logical_theta"],
        frozen_result["result"]["logical_optimal_point"],
        rtol=1.0e-9,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        [
            row["energy_after"]
            for row in canonical_result.result_payload["history"]
        ],
        [
            row["energy_after"]
            for row in frozen_result["result"]["adapt_history"]
        ],
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    accounting = canonical_result.result_payload["estimator_accounting"]
    assert accounting["N_metric"] == 0
    assert accounting["N_grad"] == sum(
        int(row["candidate_count_scored"])
        for row in canonical_result.result_payload["history"]
    )
    assert not any(
        occurrence["consumer_scope"].endswith("accepted_refit_origin")
        for occurrence in canonical_result.result_payload[
            "estimator_call_ledger"
        ]["occurrences"]
    )
