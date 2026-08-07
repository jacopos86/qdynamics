from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import pipelines.static_adapt.ra_adapt.campaign as campaign
from pipelines.static_adapt.ra_adapt.contracts import (
    RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID,
    RA_ADAPT_PROTOCOL_SCHEMA_V2,
    RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2,
    load_resolved_ra_adapt_protocol,
)
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    REPO_ROOT
    / "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_hh_ed_cutoff_reference_six_regime_20260727.json"
)
SOURCE_SHA256 = (
    "66a6409790affffd6ce8928d7fb46cc945b57d50e210d3cb215e8039a63c5573"
)
EXPECTED_ROUTE_SHA256 = (
    "04f795b0443c7a1ebcb62e9661669a765d5a0006b282f2bee043135bf390cc6b"
)


def test_persisted_prefix_rehydrates_to_exact_typed_compile_input() -> None:
    from pipelines.reporting.paper_i_run_summary import (
        PaperIAlgorithmicWork,
        PaperIPrefixCompileInput,
        PaperIPrefixOperator,
        PaperIPrefixPauliTerm,
        PaperIReferenceState,
        PaperIWorkComponents,
    )
    from pipelines.static_adapt.estimator_call_ledger import (
        projective_state_fingerprint,
    )

    fingerprint = projective_state_fingerprint((1.0 + 0.0j, 0.0 + 0.0j))
    work = PaperIAlgorithmicWork(
        components=PaperIWorkComponents(
            n_h_outer=1,
            n_h_refit=2,
            n_grad=3,
            n_metric=4,
        ),
        s_alg=10,
    )
    prefix = PaperIPrefixCompileInput(
        source_method="sr_snake",
        controller_round=1,
        active_ansatz_depth=1,
        ordered_operator_labels=("x0",),
        operators=(
            PaperIPrefixOperator(
                candidate_label="x0",
                logical_index=0,
                runtime_start=0,
                runtime_count=1,
                execution_mode="termwise_product",
                runtime_terms=(
                    PaperIPrefixPauliTerm(
                        pauli_exyz="x",
                        coefficient_real=0.0,
                        coefficient_imaginary=1.0,
                        qubit_count=1,
                    ),
                ),
            ),
        ),
        logical_parameters=(0.25,),
        runtime_parameters=(0.25,),
        reference_state=PaperIReferenceState(
            amplitudes_real=(1.0, 0.0),
            amplitudes_imaginary=(0.0, 0.0),
            qubit_count=1,
            source_label="unit_test_reference",
            state_fingerprint=fingerprint,
        ),
        checkpoint_sha256="b" * 64,
        projective_state_fingerprint=fingerprint,
        problem_request_sha256="c" * 64,
        route_profile="unit_test_route",
        route_contract_sha256=EXPECTED_ROUTE_SHA256,
        algorithmic_work=work,
    )

    persisted = json.loads(json.dumps(asdict(prefix)))
    assert campaign._prefix_from_mapping(persisted) == prefix


def _materialize(output_root: Path) -> campaign.PaperICampaignPlan:
    return campaign.materialize_paper_i_campaign(
        repository_root=REPO_ROOT,
        output_root=output_root,
        campaign_id="paper_i_ra_v2_strong_weak_macro_always_r50_test",
        run_class="candidate",
        target="local_candidate_scientific_review",
        regime_name="strong-weak",
        physics_source_path=SOURCE_PATH,
        physics_source_sha256=SOURCE_SHA256,
    )


def test_materialization_binds_exact_v2_cell_and_separate_authorization(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "candidate"
    plan = _materialize(output_root)

    assert plan.execution_authorized is False
    assert plan.run_class == "candidate"
    assert plan.ordered_cases == (
        "strong_weak__macro__always_commutation_reduced__r50",
    )
    assert plan.cell["working_cutoff"] == 3
    assert plan.cell["same_cutoff_exact_energy"] == pytest.approx(
        0.5264586847939832, abs=0.0
    )
    assert plan.cell["problem"]["problem_request_sha256"] == (
        "c85704edfe9d50742bfe8d1219033e59e354b54b51204a0a30b41e1772187860"
    )
    assert plan.cell["protocol_schema"] == RA_ADAPT_PROTOCOL_SCHEMA_V2
    assert plan.cell["algorithm_id"] == (
        RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID
    )
    assert plan.cell["route_contract_schema"] == (
        RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2
    )
    assert plan.cell["route_contract_sha256"] == EXPECTED_ROUTE_SHA256
    assert plan.cell["candidate_representation"] == "macro_generator_v1"
    assert plan.cell["active_gradient_policy"] == (
        "measured_residual_response_v1"
    )
    assert plan.cell["resource_weighting_scope"] == (
        "all_phase_resource_weighting_v1"
    )
    assert plan.cell["optimizer"] == "powell"
    assert plan.cell["optimizer_maxiter"] == 200
    assert plan.cell["seeds"] == {"adapt": 7, "transpiler": 7}
    assert plan.cell["resource_rounds"] == [50]

    protocol = load_resolved_ra_adapt_protocol(
        output_root / "resolved_protocol.json"
    )
    assert protocol.source_locks == {}
    assert protocol.parent_inventory.count == 123
    assert protocol.executable_pool.count == 102
    assert protocol.request.method.insertion.kind == (
        "always_commutation_reduced"
    )

    authorization = campaign.authorize_paper_i_campaign(
        output_root / "campaign_plan.json",
        authorization_basis="explicit_test_authorization",
        authorized_at_utc="2026-07-31T12:00:00Z",
    )
    assert plan.execution_authorized is False
    assert authorization.payload["execution_authorized"] is True
    assert authorization.payload["submission_authorized"] is False
    assert authorization.payload["accepted_state_resume_authorized"] is True
    assert authorization.payload["qiskit_observation_retry_authorized"] is True
    assert authorization.payload["protocol_sha256"] == protocol.sha256
    assert authorization.payload["route_contract_sha256"] == (
        EXPECTED_ROUTE_SHA256
    )


def test_materialization_rejects_source_hash_drift_without_creating_output(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "candidate"
    with pytest.raises(
        campaign.PaperICampaignContractError,
        match="physics source file hash drifted",
    ):
        campaign.materialize_paper_i_campaign(
            repository_root=REPO_ROOT,
            output_root=output_root,
            campaign_id="source-drift",
            run_class="candidate",
            target="local_candidate_scientific_review",
            regime_name="strong-weak",
            physics_source_path=SOURCE_PATH,
            physics_source_sha256="0" * 64,
        )
    assert not output_root.exists()


def test_authorization_rejects_coherently_rehashed_plan_tampering(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "candidate"
    _materialize(output_root)
    plan_path = output_root / "campaign_plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["cell"]["controller_horizon"] = 49
    unsigned = dict(payload)
    unsigned.pop("sha256")
    payload["sha256"] = campaign.canonical_sha256(unsigned)
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        campaign.PaperICampaignContractError,
        match="protocol drifted",
    ):
        campaign.authorize_paper_i_campaign(
            plan_path,
            authorization_basis="must_not_authorize",
        )
    assert not (output_root / "execution_authorization.json").exists()


class _FakeSummary:
    def __init__(
        self, exact_energy: float, final_energy: float, protocol: Any
    ) -> None:
        self.provenance = SimpleNamespace(
            exact_same_cutoff_energy=exact_energy,
            problem_request_sha256=(
                protocol.problem.problem_request_sha256
            ),
            route_contract_sha256=EXPECTED_ROUTE_SHA256,
            route_family="ra_adapt",
            route_profile=protocol.route_contract["route_profile"],
            candidate_representation=protocol.candidate_representation,
            reference_state_fingerprint="fake-reference-state",
            qiskit_compile_convention=(
                "table_i_basis_gate_transpile_v1"
            ),
        )
        self.accepted_error_trace = tuple(
            SimpleNamespace(
                controller_round=index,
                active_ansatz_depth=index,
                absolute_energy_error=abs(final_energy - exact_energy),
                projective_state_fingerprint="fake-terminal-state",
                checkpoint_sha256="a" * 64,
            )
            for index in range(1, 51)
        )
        work = SimpleNamespace(
            components=SimpleNamespace(
                n_h_outer=100,
                n_h_refit=200,
                n_grad=300,
                n_metric=400,
            ),
            s_alg=1000,
        )
        prefix = SimpleNamespace(
            source_method="sr_snake",
            controller_round=50,
            active_ansatz_depth=50,
            checkpoint_sha256="a" * 64,
            projective_state_fingerprint="fake-terminal-state",
            problem_request_sha256=(
                protocol.problem.problem_request_sha256
            ),
            route_profile=protocol.route_contract["route_profile"],
            route_contract_sha256=EXPECTED_ROUTE_SHA256,
            reference_state=SimpleNamespace(
                state_fingerprint="fake-reference-state"
            ),
            algorithmic_work=work,
        )
        self.requested_rounds = (
            SimpleNamespace(
                purpose="requested_controller_round",
                controller_round=50,
                active_ansatz_depth=50,
                absolute_energy_error=abs(final_energy - exact_energy),
                algorithmic_work=work,
                prefix=prefix,
                status="available",
                resources=SimpleNamespace(
                    compile_convention="table_i_basis_gate_transpile_v1",
                    compiled_two_qubit_count=123,
                    compiled_two_qubit_depth=45,
                    compiled_total_depth=67,
                ),
                failure=None,
            ),
        )
        self.canonical_all_work = work

    def to_dict(self) -> dict[str, Any]:
        def _json_ready(value: Any) -> Any:
            if isinstance(value, SimpleNamespace):
                return {
                    key: _json_ready(item)
                    for key, item in vars(value).items()
                }
            if isinstance(value, (tuple, list)):
                return [_json_ready(item) for item in value]
            return value

        return {
            "schema": "paper_i_run_summary_v1",
            "available_controller_rounds": 50,
            "accepted_error_trace": _json_ready(
                self.accepted_error_trace
            ),
            "requested_rounds": _json_ready(self.requested_rounds),
            "canonical_all_work": _json_ready(
                self.canonical_all_work
            ),
            "provenance": _json_ready(self.provenance),
        }


class _FakeResult:
    def __init__(self, protocol: Any) -> None:
        exact_energy = 0.5264586847939832
        final_energy = exact_energy + 1.0e-4
        refit = SimpleNamespace(
            initialization_policy="exact_applied_joint_step_guarded_v1",
            initialization_status="accepted",
            initialization_guard_nfev=1,
        )
        accounting = SimpleNamespace(
            complete=True,
            prefix_closure_passed=True,
            all_work=SimpleNamespace(s_alg=1000),
            raw_occurrence_total=1000,
        )
        self.protocol = protocol
        self.run = SimpleNamespace(
            stop=SimpleNamespace(completed_controller_rounds=50),
            accepted_trajectory=tuple(range(50)),
            final_state=SimpleNamespace(
                controller_round=50,
                energy=final_energy,
            ),
            estimator_accounting=accounting,
            route=SimpleNamespace(
                contract_sha256=EXPECTED_ROUTE_SHA256
            ),
            scientific_replay=tuple(
                SimpleNamespace(accepted_refit=refit) for _ in range(50)
            ),
            paper_i_summary=_FakeSummary(
                exact_energy, final_energy, protocol
            ),
        )
        self.scientific_receipts = {"verified": True}

    def to_dict(self) -> dict[str, Any]:
        final_state = {
            "controller_round": 50,
            "energy": self.run.final_state.energy,
        }
        trajectory = [
            {
                "controller_round": index,
                "energy": self.run.final_state.energy + (50 - index),
            }
            for index in range(1, 50)
        ]
        trajectory.append(final_state)
        embedded_summary = self.run.paper_i_summary.to_dict()
        embedded_summary.pop("schema")
        return {
            "schema": "paper_i_ra_adapt_result_v2",
            "protocol": self.protocol.to_dict(),
            "run": {
                "final_state": final_state,
                "accepted_trajectory": trajectory,
                "route": {
                    "family": "ra_adapt",
                    "profile": self.protocol.route_contract[
                        "route_profile"
                    ],
                    "contract_sha256": EXPECTED_ROUTE_SHA256,
                },
                "stop": {"completed_controller_rounds": 50},
                "estimator_accounting": {
                    "all_work": {"s_alg": 1000}
                },
                "paper_i_summary": embedded_summary,
            },
        }


def _write_fake_checkpoint(protocol: Any, *, history_count: int) -> None:
    checkpoint = protocol.request.observation.checkpoint
    ledger = protocol.request.observation.estimator_ledger
    assert checkpoint is not None
    assert ledger is not None
    ledger_sidecar = checkpoint.path.with_name("ledger-sidecar.json")
    resume_sidecar = checkpoint.path.with_name("resume-sidecar.json")
    ledger_sidecar.write_text("{}\n", encoding="utf-8")
    resume_sidecar.write_text("{}\n", encoding="utf-8")
    checkpoint.path.write_text(
        json.dumps(
            {
                "checkpoint": {
                    "estimator_call_ledger_checkpoint": {
                        "path": ledger_sidecar.name,
                        "sha256": hashlib.sha256(
                            ledger_sidecar.read_bytes()
                        ).hexdigest(),
                    }
                },
                "adapt_vqe": {
                    "history_count": history_count,
                    "verified_singleton_resume_sidecar": {
                        "path": resume_sidecar.name,
                        "sha256": hashlib.sha256(
                            resume_sidecar.read_bytes()
                        ).hexdigest(),
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_execute_uses_only_bound_plan_and_emits_terminal_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "candidate"
    _materialize(output_root)
    campaign.authorize_paper_i_campaign(
        output_root / "campaign_plan.json",
        authorization_basis="explicit_test_authorization",
        authorized_at_utc="2026-07-31T12:00:00Z",
    )

    def _fake_execute(
        _problem: Any,
        protocol: Any,
        *,
        resume_checkpoint: Path | None,
    ) -> _FakeResult:
        assert resume_checkpoint is None
        checkpoint = protocol.request.observation.checkpoint
        ledger = protocol.request.observation.estimator_ledger
        assert checkpoint is not None
        assert ledger is not None
        _write_fake_checkpoint(protocol, history_count=50)
        ledger.path.write_text("{}\n", encoding="utf-8")
        return _FakeResult(protocol)

    monkeypatch.setattr(campaign, "_execute_scientific", _fake_execute)
    terminal = campaign.execute_paper_i_campaign(
        output_root / "campaign_plan.json",
        output_root / "execution_authorization.json",
    )

    assert terminal["status"] == "passed"
    assert terminal["accepted_controller_rounds"] == 50
    assert terminal["round_50_qiskit_resources"] == {
        "compile_convention": "table_i_basis_gate_transpile_v1",
        "compiled_two_qubit_count": 123,
        "compiled_two_qubit_depth": 45,
        "compiled_total_depth": 67,
    }
    for name in (
        "run_manifest.json",
        "checkpoint.current.json",
        "estimator_ledger.json",
        "result.json",
        "summary.json",
        "scientific_receipts.json",
        "validation.json",
        "terminal_receipt.json",
    ):
        assert (output_root / name).is_file()
    assert not (output_root / "failure_receipt.json").exists()


def test_execute_rejects_authorization_drift_before_scientific_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "candidate"
    _materialize(output_root)
    campaign.authorize_paper_i_campaign(
        output_root / "campaign_plan.json",
        authorization_basis="explicit_test_authorization",
        authorized_at_utc="2026-07-31T12:00:00Z",
    )
    authorization_path = output_root / "execution_authorization.json"
    payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    payload["protocol_sha256"] = "f" * 64
    unsigned = dict(payload)
    unsigned.pop("sha256")
    payload["sha256"] = campaign.canonical_sha256(unsigned)
    authorization_path.write_text(json.dumps(payload), encoding="utf-8")

    def _must_not_run(
        _problem: Any,
        _protocol: Any,
        *,
        resume_checkpoint: Path | None,
    ) -> Any:
        raise AssertionError("scientific execution must not start")

    monkeypatch.setattr(campaign, "_execute_scientific", _must_not_run)
    with pytest.raises(
        campaign.PaperICampaignContractError,
        match="Authorization drifted",
    ):
        campaign.execute_paper_i_campaign(
            output_root / "campaign_plan.json",
            authorization_path,
        )
    failure = json.loads(
        (output_root / "failure_receipt.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert not (output_root / "run_manifest.json").exists()


def test_validation_rejects_wrong_round50_qiskit_convention(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "candidate"
    plan = _materialize(output_root)
    protocol = load_resolved_ra_adapt_protocol(
        output_root / "resolved_protocol.json"
    )
    result = _FakeResult(protocol)
    result.run.paper_i_summary.requested_rounds[
        0
    ].resources.compile_convention = "wrong_transpile_contract"

    with pytest.raises(
        campaign.PaperICampaignContractError,
        match="locked compiler contract",
    ):
        campaign._validate_completed_result(
            result,
            plan=plan,
            protocol=protocol,
        )


def test_qiskit_observation_retry_is_additive_and_repeatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pipelines.reporting.paper_i_run_summary import (
        PaperIAlgorithmicWork,
        PaperIWorkComponents,
    )

    output_root = tmp_path / "candidate"
    _materialize(output_root)
    campaign.authorize_paper_i_campaign(
        output_root / "campaign_plan.json",
        authorization_basis="explicit_test_authorization",
        authorized_at_utc="2026-07-31T12:00:00Z",
    )

    def _fake_execute(
        _problem: Any,
        protocol: Any,
        *,
        resume_checkpoint: Path | None,
    ) -> _FakeResult:
        assert resume_checkpoint is None
        _write_fake_checkpoint(protocol, history_count=50)
        ledger = protocol.request.observation.estimator_ledger
        assert ledger is not None
        ledger.path.write_text("{}\n", encoding="utf-8")
        result = _FakeResult(protocol)
        observation = result.run.paper_i_summary.requested_rounds[0]
        observation.status = "retryable_tooling_error"
        observation.resources = None
        observation.failure = SimpleNamespace(
            exception_type="ImportError",
            message="simulated Qiskit tooling failure",
            retryable=True,
        )
        return result

    monkeypatch.setattr(campaign, "_execute_scientific", _fake_execute)
    terminal = campaign.execute_paper_i_campaign(
        output_root / "campaign_plan.json",
        output_root / "execution_authorization.json",
    )
    assert terminal["status"] == (
        "scientific_complete_retryable_observation_failure"
    )
    immutable_paths = tuple(
        output_root / name
        for name in (
            "campaign_plan.json",
            "resolved_protocol.json",
            "runtime_source_inventory.json",
            "execution_authorization.json",
            "run_manifest.json",
            "checkpoint.current.json",
            "estimator_ledger.json",
            "result.json",
            "summary.json",
            "scientific_receipts.json",
            "validation.json",
            "terminal_receipt.json",
            "ledger-sidecar.json",
            "resume-sidecar.json",
        )
    )
    immutable_bytes = {
        path: path.read_bytes() for path in immutable_paths
    }
    summary = json.loads(
        (output_root / "summary.json").read_text(encoding="utf-8")
    )
    prefix_payload = summary["requested_rounds"][0]["prefix"]
    work = PaperIAlgorithmicWork(
        components=PaperIWorkComponents(
            n_h_outer=100,
            n_h_refit=200,
            n_grad=300,
            n_metric=400,
        ),
        s_alg=1000,
    )
    fake_prefix = SimpleNamespace(
        source_method="sr_snake",
        controller_round=50,
        active_ansatz_depth=50,
        checkpoint_sha256="a" * 64,
        projective_state_fingerprint="fake-terminal-state",
        problem_request_sha256=prefix_payload[
            "problem_request_sha256"
        ],
        route_profile=prefix_payload["route_profile"],
        route_contract_sha256=EXPECTED_ROUTE_SHA256,
        reference_state=SimpleNamespace(
            qubit_count=8,
            state_fingerprint="fake-reference-state",
        ),
        runtime_parameters=tuple(0.0 for _ in range(50)),
        algorithmic_work=work,
    )
    monkeypatch.setattr(
        campaign,
        "_prefix_from_mapping",
        lambda observed: (
            fake_prefix
            if dict(observed) == prefix_payload
            else pytest.fail("retry reconstructed the wrong prefix")
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_execute_scientific",
        lambda *_args, **_kwargs: pytest.fail(
            "Qiskit observation retry invoked the scientific engine"
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_source_inventory",
        lambda *_args, **_kwargs: pytest.fail(
            "Qiskit-only retry required current scientific-source equality"
        ),
    )
    attempts = 0

    def _compile(_prefix: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        assert _prefix is fake_prefix
        if attempts <= 2:
            raise RuntimeError("simulated retry tooling failure")
        return {
            "compiled_circuit_stats_status": "ok",
            "first_hit_cost_source_kind": (
                "canonical_paper_i_accepted_prefix"
            ),
            "compiled_resource_source_kind": (
                "canonical_paper_i_accepted_prefix"
            ),
            "compiled_resource_qiskit_validated": True,
            "qiskit_first_hit_cost_validated": False,
            "compiled_basis_gates": [
                "id",
                "x",
                "sx",
                "rx",
                "ry",
                "rz",
                "h",
                "s",
                "sdg",
                "cx",
                "cz",
            ],
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "qiskit_transpile_optimization_level": 0,
            "qiskit_transpile_seed": 7,
            "grouped_exact_coefficient_tolerance": 1.0e-12,
            "grouped_exact_max_active_qubits": 5,
            "angle_convention": (
                "structural_nonzero_placeholder_angles_v1"
            ),
            "compiled_circuit_scope": (
                "ansatz_circuit_including_reference_state"
            ),
            "num_qubits": 8,
            "logical_operator_count": 50,
            "runtime_rotation_count": 50,
            "compiled_count_1q_total": 654,
            "compiled_count_2q_total": 321,
            "compiled_depth_2q_total": 54,
            "compiled_depth_total": 76,
        }

    monkeypatch.setattr(campaign, "_compile_qiskit_prefix", _compile)
    with pytest.raises(
        RuntimeError, match="simulated retry tooling failure"
    ):
        campaign.retry_paper_i_campaign_qiskit_observation(
            output_root / "campaign_plan.json",
            output_root / "execution_authorization.json",
        )
    first_failure = (
        output_root
        / "qiskit_observation_retries/attempt_001.failure.json"
    )
    assert first_failure.is_file()
    authentic_failure = first_failure.read_bytes()
    malformed = json.loads(authentic_failure)
    malformed["status"] = "available"
    malformed_unsigned = dict(malformed)
    malformed_unsigned.pop("sha256")
    malformed["sha256"] = campaign.canonical_sha256(
        malformed_unsigned
    )
    first_failure.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(
        campaign.PaperICampaignContractError,
        match="Failed Qiskit retry history is malformed",
    ):
        campaign.retry_paper_i_campaign_qiskit_observation(
            output_root / "campaign_plan.json",
            output_root / "execution_authorization.json",
        )
    assert attempts == 1
    first_failure.write_bytes(authentic_failure)

    with pytest.raises(
        RuntimeError, match="simulated retry tooling failure"
    ):
        campaign.retry_paper_i_campaign_qiskit_observation(
            output_root / "campaign_plan.json",
            output_root / "execution_authorization.json",
        )
    assert (
        output_root
        / "qiskit_observation_retries/attempt_002.failure.json"
    ).is_file()

    receipt = campaign.retry_paper_i_campaign_qiskit_observation(
        output_root / "campaign_plan.json",
        output_root / "execution_authorization.json",
    )
    assert receipt["status"] == "available"
    assert receipt["resources"] == {
        "compile_convention": "table_i_basis_gate_transpile_v1",
        "compiled_two_qubit_count": 321,
        "compiled_two_qubit_depth": 54,
        "compiled_total_depth": 76,
    }
    assert (
        output_root
        / "qiskit_observation_retries/attempt_003.result.json"
    ).is_file()
    assert {
        path: path.read_bytes() for path in immutable_paths
    } == immutable_bytes
    with pytest.raises(
        campaign.PaperICampaignContractError,
        match="successful round-50 Qiskit retry already exists",
    ):
        campaign.retry_paper_i_campaign_qiskit_observation(
            output_root / "campaign_plan.json",
            output_root / "execution_authorization.json",
        )
    assert attempts == 3


def test_authenticated_resume_preserves_original_attempt_and_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "candidate"
    _materialize(output_root)
    campaign.authorize_paper_i_campaign(
        output_root / "campaign_plan.json",
        authorization_basis="explicit_test_authorization",
        authorized_at_utc="2026-07-31T12:00:00Z",
    )
    protocol = load_resolved_ra_adapt_protocol(
        output_root / "resolved_protocol.json"
    )

    def _interrupt_after_two(
        _problem: Any,
        observed_protocol: Any,
        *,
        resume_checkpoint: Path | None,
    ) -> Any:
        assert resume_checkpoint is None
        _write_fake_checkpoint(observed_protocol, history_count=2)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        campaign, "_execute_scientific", _interrupt_after_two
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        campaign.execute_paper_i_campaign(
            output_root / "campaign_plan.json",
            output_root / "execution_authorization.json",
        )
    checkpoint_path = output_root / "checkpoint.current.json"
    authenticated_checkpoint = checkpoint_path.read_bytes()
    _write_fake_checkpoint(protocol, history_count=3)
    with pytest.raises(
        campaign.PaperICampaignContractError,
        match="not the checkpoint bound",
    ):
        campaign.execute_paper_i_campaign(
            output_root / "campaign_plan.json",
            output_root / "execution_authorization.json",
            resume=True,
        )
    checkpoint_path.write_bytes(authenticated_checkpoint)

    def _interrupt_first_resume(
        _problem: Any,
        observed_protocol: Any,
        *,
        resume_checkpoint: Path | None,
    ) -> Any:
        assert resume_checkpoint == output_root / "checkpoint.current.json"
        _write_fake_checkpoint(observed_protocol, history_count=3)
        raise RuntimeError("simulated resume interruption")

    monkeypatch.setattr(
        campaign, "_execute_scientific", _interrupt_first_resume
    )
    with pytest.raises(
        RuntimeError, match="simulated resume interruption"
    ):
        campaign.execute_paper_i_campaign(
            output_root / "campaign_plan.json",
            output_root / "execution_authorization.json",
            resume=True,
        )

    def _fake_resume(
        _problem: Any,
        observed_protocol: Any,
        *,
        resume_checkpoint: Path | None,
    ) -> _FakeResult:
        assert resume_checkpoint == output_root / "checkpoint.current.json"
        _write_fake_checkpoint(observed_protocol, history_count=50)
        ledger = observed_protocol.request.observation.estimator_ledger
        assert ledger is not None
        ledger.path.write_text("{}\n", encoding="utf-8")
        return _FakeResult(observed_protocol)

    monkeypatch.setattr(campaign, "_execute_scientific", _fake_resume)
    terminal = campaign.execute_paper_i_campaign(
        output_root / "campaign_plan.json",
        output_root / "execution_authorization.json",
        resume=True,
    )

    assert terminal["status"] == "passed"
    assert (output_root / "run_manifest.json").is_file()
    assert (output_root / "failure_receipt.json").is_file()
    assert (
        output_root / "resume_attempts/attempt_002.manifest.json"
    ).is_file()
    assert (
        output_root / "resume_attempts/attempt_001.failure.json"
    ).is_file()
    assert not (
        output_root / "resume_attempts/attempt_002.failure.json"
    ).exists()
