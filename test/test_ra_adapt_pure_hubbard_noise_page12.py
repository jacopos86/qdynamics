from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import pytest

from pipelines.static_adapt import adapt_pipeline as adapt_pipeline_module
from pipelines.static_adapt.adapt_pipeline import (
    _insertion_commutation_plateau_round_policy,
)
from pipelines.static_adapt.ra_adapt import (
    RAAdaptOperationalControls,
    RAAdaptRequest,
    build_paper_i_pure_hubbard_noise_page12_problem,
    build_paper_i_pure_hubbard_noise_page12_request,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_STATIONARY,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    RA_ADAPT_PROTOCOL_SCHEMA_V1,
    RA_STAGED_SELECTOR_ID,
    RESOURCE_WEIGHTING_ALL_PHASE,
    _attach_validated_bundle_protocol_authority,
    _mint_bundle_protocol_materialization_authority,
    bundle_protocol_materialization_receipt,
    resolved_ra_adapt_protocol_from_mapping,
)
from pipelines.static_adapt.ra_adapt.engine import (
    _repaired_route_contract,
    build_resolved_ra_protocol,
    run_ra_adapt,
)
from pipelines.static_adapt.ra_adapt.pools import (
    build_guarded_single_pauli_pool,
)
from pipelines.static_adapt.ra_adapt.pure_hubbard_noise_page12 import (
    PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ADAPTER_ID,
    PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
    PAPER_I_PURE_HUBBARD_NOISE_PAGE12_SOURCE_LOCK_KEY,
    PaperIPureHubbardNoisePage12CandidateAdapter,
    paper_i_pure_hubbard_noise_page12_application_source_contract,
    pure_hubbard_noise_level_contract,
)
from pipelines.static_adapt.sr_snake.contracts import AcceptedStateResume
from pipelines.static_adapt.sr_snake_route_profile import (
    canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256,
)


def _authority(source_sha256: str, *, protocol_sha256: str | None = None):
    source_locks = {
        "source_locks_manifest_sha256": "1" * 64,
        "implementation_source_inventory_sha256": "2" * 64,
        "cell_source_lock_id": "paper_i_pure_hubbard_noise_page12_test",
        "cell_source_lock_sha256": "3" * 64,
        "ed_cutoff_reference_sha256": "4" * 64,
        PAPER_I_PURE_HUBBARD_NOISE_PAGE12_SOURCE_LOCK_KEY: source_sha256,
    }
    receipt = bundle_protocol_materialization_receipt(
        bundle_id="paper_i_pure_hubbard_noise_page12_test_bundle",
        bundle_manifest_sha256="5" * 64,
        source_locks_sha256="1" * 64,
        source_lock_refs=source_locks,
        cell_id="paper_i_pure_hubbard_noise_page12_test",
        source_lock_id="paper_i_pure_hubbard_noise_page12_test",
        protocol_schema=RA_ADAPT_PROTOCOL_SCHEMA_V1,
        algorithm_id=PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
        candidate_representation=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        selector_identity=RA_STAGED_SELECTOR_ID,
        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )
    return _mint_bundle_protocol_materialization_authority(
        receipt,
        source_lock_refs=source_locks,
        protocol_sha256=protocol_sha256,
    )


def _count_nested_key(value, key: str) -> int:
    if isinstance(value, dict):
        return int(key in value) + sum(
            _count_nested_key(child, key) for child in value.values()
        )
    if isinstance(value, list):
        return sum(_count_nested_key(child, key) for child in value)
    return 0


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("low", (1.0e-6, 1.0e-8, 1.0e-7, 2.0e-4, 6.0e-4)),
        (
            "high",
            (7.071067811865475e-5, 1.0e-6, 1.0e-5, 2.0e-3, 6.0e-3),
        ),
        ("extreme", (1.0e-2, 1.0e-3, 1.0e-2, 6.0e-2, 6.0e-2)),
    ],
)
def test_pure_hubbard_noise_levels_are_fixed_and_iid(
    level: str,
    expected: tuple[float, ...],
) -> None:
    contract = pure_hubbard_noise_level_contract(level)

    assert tuple(contract["noise_tuple"]) == expected
    assert contract["noise_tuple_order"] == [
        "sigma_E",
        "p1",
        "p2",
        "epsilon1",
        "epsilon2",
    ]
    assert contract["value_noise"] == {
        "model": "gaussian_iid_v1",
        "std": expected[0],
        "seed": 702688422,
        "frozen_keyed": False,
        "semantic": "post_expectation_value_noise_not_physical_shots",
        "std_source": "explicit_std",
        "physical_shots_unchanged": True,
        "fixed_gate_error_reduction_claimed": False,
    }
    assert contract["synthetic_coherent_seed"] == 20260609
    assert contract["optimizer_evaluation_order"] == "serial_v1"


@pytest.mark.parametrize("u", [1.5, 8.0])
def test_named_pure_hubbard_problem_and_ordinary_guards(u: float) -> None:
    problem = build_paper_i_pure_hubbard_noise_page12_problem(u=u)

    assert problem.family_key == "hubbard"
    assert problem.request.num_sites == 2
    assert problem.sector.num_particles == (1, 1)
    assert problem.layout.fermion_qubits == 4
    assert problem.layout.boson_qubits == 0
    assert problem.layout.total_qubits == 4
    with pytest.raises(ValueError, match="Paper-I HH L=2|staged RA singleton"):
        build_guarded_single_pauli_pool(problem)
    with pytest.raises(ValueError, match="Hubbard--Holstein L=2"):
        run_ra_adapt(problem, RAAdaptRequest())

    drifted = replace(problem, request=replace(problem.request, u=2.0))
    named = build_paper_i_pure_hubbard_noise_page12_request(
        noise_level="low",
        maximum_controller_rounds=1,
    )
    with pytest.raises(ValueError, match="U/t"):
        named.adapter.executable_pool(drifted)


@pytest.mark.parametrize(
    ("u", "expected_u_rational"),
    [(1.5, (3, 2)), (8.0, (8, 1))],
)
def test_application_source_digest_uses_validated_analytic_ed_identity(
    u: float,
    expected_u_rational: tuple[int, int],
) -> None:
    problem = build_paper_i_pure_hubbard_noise_page12_problem(u=u)
    request = build_paper_i_pure_hubbard_noise_page12_request(
        noise_level="low",
        maximum_controller_rounds=1,
    )
    observed_energy = float(problem.exact_target.resolve_energy())
    lower_energy = math.nextafter(observed_energy, -math.inf)
    upper_energy = math.nextafter(observed_energy, math.inf)
    assert lower_energy != upper_energy
    lower_problem = replace(
        problem,
        exact_target=replace(
            problem.exact_target,
            resolve_energy=lambda **_: lower_energy,
        ),
    )
    upper_problem = replace(
        problem,
        exact_target=replace(
            problem.exact_target,
            resolve_energy=lambda **_: upper_energy,
        ),
    )

    lower = paper_i_pure_hubbard_noise_page12_application_source_contract(
        lower_problem,
        request,
    )
    upper = paper_i_pure_hubbard_noise_page12_application_source_contract(
        upper_problem,
        request,
    )

    assert lower["schema"] == (
        "paper_i_pure_hubbard_noise_page12_application_source_contract_v2"
    )
    assert "energy" not in lower["same_cutoff_exact_reference"]
    assert lower["same_cutoff_exact_reference"]["evaluation_policy_id"] == (
        "runtime_same_cutoff_exact_diagnostic_full_precision_v1"
    )
    assert lower["same_cutoff_exact_reference"]["analytic_reference"] == {
        "formula_id": "l2_open_half_filled_hubbard_ground_energy_v1",
        "t": {"numerator": 1, "denominator": 1},
        "U": {
            "numerator": expected_u_rational[0],
            "denominator": expected_u_rational[1],
        },
    }
    assert lower["same_cutoff_exact_reference"]["controller_input"] is False
    assert lower["same_cutoff_exact_reference"] == upper[
        "same_cutoff_exact_reference"
    ]
    assert lower["sha256"] == upper["sha256"]

    wrong_problem = replace(
        problem,
        exact_target=replace(
            problem.exact_target,
            resolve_energy=lambda **_: observed_energy + 1.0e-6,
        ),
    )
    with pytest.raises(ValueError, match="same-cutoff ED reference drifted"):
        paper_i_pure_hubbard_noise_page12_application_source_contract(
            wrong_problem,
            request,
        )


def test_named_noise_protocol_binds_full_noise_and_roundtrips() -> None:
    problem = build_paper_i_pure_hubbard_noise_page12_problem(u=1.5)
    request = build_paper_i_pure_hubbard_noise_page12_request(
        noise_level="high",
        maximum_controller_rounds=3,
    )
    source = paper_i_pure_hubbard_noise_page12_application_source_contract(
        problem,
        request,
    )

    with pytest.raises(ValueError, match="application source-lock digest"):
        build_resolved_ra_protocol(
            problem,
            request,
            materialization_authority=_authority("f" * 64),
        )

    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=_authority(source["sha256"]),
    )
    assert protocol.algorithm_id == PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID
    assert protocol.adapter_id == PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ADAPTER_ID
    assert protocol.horizon == 3
    assert protocol.active_gradient_policy == ACTIVE_GRADIENT_STATIONARY
    assert protocol.resource_weighting_scope == RESOURCE_WEIGHTING_ALL_PHASE
    route = protocol.route_contract
    assert route is not None
    noise = route["execution_settings"]["ra_controller_noise_contract"]
    assert noise["surface"] == {
        "candidate_gradient_scoring": "noisy",
        "powell_refit_objective": "noisy",
        "geometry_and_gram": "exact",
        "reported_energy": "exact_diagnostic",
    }
    assert noise["optimizer_evaluation_order"] == "serial_v1"
    assert route["execution_settings"]["adapt_parallel_gradient_workers"] == 1
    assert route["semantic_invariants"]["physical_operator_lanes_active"] is False
    assert route["semantic_invariants"]["phase_ii_compile_cost_source"] == "backend_transpile_v1"
    assert route["semantic_invariants"]["phase_iii_compile_cost_source"] == "backend_transpile_v1"
    invariants = route["semantic_invariants"]
    assert "plateau_prior_mean_decrease_ratio_threshold" not in invariants
    assert invariants["experimental_insertion_policy"] == (
        "insertion_commutation_plateau_v1"
    )
    assert invariants["plateau_progress_statistic"] == (
        "marginal_to_prior_cumulative_energy_decrease_v1"
    )
    assert invariants["plateau_cumulative_decrease_ratio_threshold"] == 1.0e-4
    assert invariants["plateau_threshold_comparison"] == (
        "marginal_to_prior_cumulative_strictly_below_v1"
    )
    assert invariants["plateau_energy_source"] == (
        "persisted_noisy_controller_energy_before_after_v1"
    )
    assert source["noise"] == {
        key: value for key, value in noise.items() if key != "surface"
    }
    assert source["scientific_settings"]["plateau_energy_source"] == (
        "persisted_noisy_controller_energy_before_after_v1"
    )

    restored = resolved_ra_adapt_protocol_from_mapping(protocol.to_dict())
    assert restored.sha256 == protocol.sha256
    assert isinstance(
        restored.request.adapter,
        PaperIPureHubbardNoisePage12CandidateAdapter,
    )


def test_named_noise_contract_tamper_fails_materialization() -> None:
    problem = build_paper_i_pure_hubbard_noise_page12_problem(u=8.0)
    request = build_paper_i_pure_hubbard_noise_page12_request(
        noise_level="extreme",
        maximum_controller_rounds=1,
    )
    source = paper_i_pure_hubbard_noise_page12_application_source_contract(
        problem,
        request,
    )
    tampered = replace(
        request,
        adapter=replace(request.adapter, noise_level_id="low"),
    )

    with pytest.raises(ValueError, match="source-lock|noise"):
        build_resolved_ra_protocol(
            problem,
            tampered,
            materialization_authority=_authority(source["sha256"]),
        )


def test_named_v1_plateau_uses_persisted_noisy_controller_energy() -> None:
    history = [
        {
            "energy_before_opt": 0.0,
            "energy_after_opt": -10.0,
            "controller_noise": {
                "controller_energy_before": 0.0,
                "controller_energy_after": -1.0,
            },
        },
        {
            "energy_before_opt": -10.0,
            "energy_after_opt": -11.0,
            "controller_noise": {
                "controller_energy_before": -1.0,
                "controller_energy_after": -1.00005,
            },
        },
    ]

    exact_diagnostic = _insertion_commutation_plateau_round_policy(
        history=history,
        policy="insertion_commutation_plateau_v1",
    )
    noisy_controller = _insertion_commutation_plateau_round_policy(
        history=history,
        policy="insertion_commutation_plateau_v1",
        controller_noise_energy_source=True,
    )

    assert exact_diagnostic["domain_open"] is False
    assert noisy_controller["domain_open"] is True
    assert noisy_controller["energy_track"] == "noisy_controller"
    assert noisy_controller["trigger_source"] == (
        "persisted_noisy_controller_energy_before_after_v1"
    )
    assert noisy_controller[
        "marginal_to_prior_cumulative_decrease_ratio"
    ] == pytest.approx(5.0e-5)
    assert noisy_controller["cumulative_decrease_ratio_threshold"] == 1.0e-4


def test_named_pure_hubbard_noise_executes_one_real_round(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    problem = build_paper_i_pure_hubbard_noise_page12_problem(u=1.5)
    request = build_paper_i_pure_hubbard_noise_page12_request(
        noise_level="low",
        maximum_controller_rounds=1,
        output_dir=tmp_path,
    )
    source = paper_i_pure_hubbard_noise_page12_application_source_contract(
        problem,
        request,
    )
    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=_authority(source["sha256"]),
    )
    protocol = _attach_validated_bundle_protocol_authority(
        protocol,
        _authority(source["sha256"], protocol_sha256=protocol.sha256),
    )

    result = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=1,
        ),
    )

    assert result.run.stop.completed_controller_rounds == 1
    assert len(result.run.accepted_trajectory) == 1
    row = result.run.accepted_trajectory[0]
    receipt = result.scientific_receipts["controller_noise"]
    assert receipt["final_controller_energy"] != row.energy
    assert receipt["final_exact_diagnostic_energy"] == row.energy
    assert receipt["candidate_gradient_scoring"] == "noisy"
    assert receipt["powell_refit_objective"] == "noisy"
    assert receipt["geometry_and_gram"] == "exact"
    assert receipt["value_noise"]["draw_count"] > 0
    assert result.run.paper_i_summary is not None

    records = receipt["evaluation_records"]
    assert len(records) == receipt["evaluation_count"]
    assert [record["evaluation_ordinal"] for record in records] == list(
        range(1, len(records) + 1)
    )
    assert any(
        "phase0_phase1" in record["consumer_scope"]
        and record["probe_sign"] == "plus"
        for record in records
    )
    assert any(
        "phase0_phase1" in record["consumer_scope"]
        and record["probe_sign"] == "minus"
        for record in records
    )
    assert any(
        record["stage"] == "phase2"
        and record["probe_sign"] == "plus"
        for record in records
    )
    assert any(
        record["stage"] == "phase2"
        and record["probe_sign"] == "minus"
        for record in records
    )
    assert any(record["stage"] == "depth_opt" for record in records)
    assert any(
        record["stage"] == "accepted_refit_same_circuit_incumbent"
        for record in records
    )
    assert all(
        int(compiled["synthetic_coherent"]["inserted_count"]) > 0
        for compiled in receipt["compiled_noise_receipts"].values()
    )
    round_receipt = result.scientific_receipts[
        "accepted_round_receipts"
    ][0]
    transition = round_receipt["controller_noise"]
    assert transition["controller_energy_after"] == receipt[
        "final_controller_energy"
    ]
    assert transition["exact_diagnostic_energy_after"] == row.energy
    assert "runtime_checkpoint" not in transition
    delta = transition["runtime_delta"]
    assert delta["schema"] == (
        "paper_i_pure_hubbard_controller_noise_transition_delta_v1"
    )
    assert delta["evaluation_count_before"] == 0
    assert delta["evaluation_count_after"] == receipt["evaluation_count"]
    assert len(delta["evaluation_records_delta"]) == receipt[
        "evaluation_count"
    ]
    assert delta["rng_state_after"]["draw_count"] == receipt["value_noise"][
        "draw_count"
    ]

    # Reporting authenticates the fixed noise rung through the exact route
    # digest.  All three settled rungs must therefore reconstruct, while the
    # named route must inherit the canonical cumulative-relative v1 parent.
    from pipelines.reporting.paper_i_run_summary import (
        _validate_canonical_identity,
    )

    expected_parent_sha256 = (
        canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256()
    )
    for noise_level_id in ("low", "high", "extreme"):
        rung_request = build_paper_i_pure_hubbard_noise_page12_request(
            noise_level=noise_level_id,
            maximum_controller_rounds=1,
        )
        (
            rung_profile_request,
            rung_profile,
            rung_contract,
            rung_contract_sha256,
        ) = _repaired_route_contract(
            rung_request,
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
            algorithm_id=PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
            problem=problem,
        )
        lineage = rung_contract["lineage_authority"]
        assert lineage["parent_contract_sha256"] == expected_parent_sha256
        assert lineage["parent_route_profile"].endswith(
            "insertion_commutation_plateau_v1"
        )
        rung_run = replace(
            result.run,
            route=replace(
                result.run.route,
                family=str(rung_contract["route_family"]),
                profile_request=rung_profile_request,
                profile=rung_profile,
                contract_sha256=rung_contract_sha256,
                method=rung_request.method,
            ),
        )
        _validate_canonical_identity(rung_run)


def test_named_noise_route_fails_closed_when_candidate_cache_is_not_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", raising=False)
    problem = build_paper_i_pure_hubbard_noise_page12_problem(u=1.5)
    request = build_paper_i_pure_hubbard_noise_page12_request(
        noise_level="low",
        maximum_controller_rounds=1,
    )
    source = paper_i_pure_hubbard_noise_page12_application_source_contract(
        problem,
        request,
    )
    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=_authority(source["sha256"]),
    )
    protocol = _attach_validated_bundle_protocol_authority(
        protocol,
        _authority(source["sha256"], protocol_sha256=protocol.sha256),
    )

    with pytest.raises(
        ValueError,
        match="STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off",
    ):
        run_ra_adapt(
            problem,
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=1,
            ),
        )


def test_named_noise_resume_restores_rng_cursor_before_new_evaluation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    live_history_controller_noise_schemas: list[tuple[str, ...]] = []
    retained_projected_round_controller_noise: list[
        tuple[tuple[str, int], ...]
    ] = []
    original_write_current = (
        adapt_pipeline_module._DefaultNoPruneNumericalSession
        ._write_current_checkpoint
    )
    original_finalize = (
        adapt_pipeline_module._DefaultNoPruneNumericalSession.finalize
    )

    def observe_live_history(self, *args, **kwargs):
        live_history_controller_noise_schemas.append(
            tuple(
                str(
                    row["active_prefix_checkpoint"]["controller_noise"][
                        "schema"
                    ]
                )
                for row in self.cursor.history
            )
        )
        return original_write_current(self, *args, **kwargs)

    monkeypatch.setattr(
        adapt_pipeline_module._DefaultNoPruneNumericalSession,
        "_write_current_checkpoint",
        observe_live_history,
    )

    def observe_retained_projected_rounds(self, *args, **kwargs):
        retained_projected_round_controller_noise.append(
            tuple(
                (
                    str(
                        checkpoint["controller_noise"]["schema"]
                    ),
                    _count_nested_key(checkpoint, "evaluation_records"),
                )
                for checkpoint in (
                    projected.checkpoint_projection.record.to_mutable_mapping()
                    for projected in kwargs["projected_rounds"]
                )
            )
        )
        return original_finalize(self, *args, **kwargs)

    monkeypatch.setattr(
        adapt_pipeline_module._DefaultNoPruneNumericalSession,
        "finalize",
        observe_retained_projected_rounds,
    )
    problem = build_paper_i_pure_hubbard_noise_page12_problem(u=1.5)
    request = build_paper_i_pure_hubbard_noise_page12_request(
        noise_level="low",
        maximum_controller_rounds=2,
        output_dir=tmp_path,
    )
    source = paper_i_pure_hubbard_noise_page12_application_source_contract(
        problem,
        request,
    )
    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=_authority(source["sha256"]),
    )
    protocol = _attach_validated_bundle_protocol_authority(
        protocol,
        _authority(source["sha256"], protocol_sha256=protocol.sha256),
    )
    first = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=1,
            observation=request.observation,
        ),
    )
    first_noise = first.scientific_receipts["controller_noise"]
    checkpoint = tmp_path / "current.json"
    assert checkpoint.is_file()
    checkpoint_bytes = checkpoint.read_bytes()
    tampered_checkpoint = json.loads(checkpoint_bytes)
    assert _count_nested_key(
        tampered_checkpoint,
        "evaluation_records",
    ) == 1
    adapt = tampered_checkpoint["adapt_vqe"]
    history_controller = adapt["history"][-1][
        "active_prefix_checkpoint"
    ]["controller_noise"]
    assert history_controller["schema"] == (
        "paper_i_pure_hubbard_controller_noise_checkpoint_binding_v1"
    )
    terminal = adapt["terminal_active_prefix_checkpoint"]
    tampered_controller = terminal["controller_noise"]
    assert tampered_controller["schema"] == (
        "paper_i_pure_hubbard_controller_noise_checkpoint_v1"
    )
    assert adapt["continuation"]["terminal_active_prefix_checkpoint"] == {
        "schema": "paper_i_signed_active_prefix_checkpoint_binding_v1",
        "checkpoint_sha256": terminal["checkpoint_sha256"],
    }
    tampered_controller["rng_state"]["draw_count"] += 1
    tampered_controller["evaluation_records"][0]["mean"] += 1.0
    checkpoint.write_text(
        json.dumps(tampered_checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered_resume = AcceptedStateResume(
        checkpoint_path=checkpoint,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="digest|sidecar|projection|checkpoint"):
        run_ra_adapt(
            problem,
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=2,
                resume=tampered_resume,
                observation=request.observation,
            ),
        )
    checkpoint.write_bytes(checkpoint_bytes)
    resume = AcceptedStateResume(
        checkpoint_path=checkpoint,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )

    continued = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=2,
            resume=resume,
            observation=request.observation,
        ),
    )
    continued_noise = continued.scientific_receipts["controller_noise"]
    first_count = int(first_noise["evaluation_count"])
    continued_records = continued_noise["evaluation_records"]
    continued_checkpoint = json.loads(checkpoint.read_bytes())

    assert continued.run.stop.completed_controller_rounds == 2
    assert _count_nested_key(
        continued_checkpoint,
        "evaluation_records",
    ) == 1
    assert all(
        row["active_prefix_checkpoint"]["controller_noise"]["schema"]
        == "paper_i_pure_hubbard_controller_noise_checkpoint_binding_v1"
        for row in continued_checkpoint["adapt_vqe"]["history"]
    )
    assert continued_records[:first_count] == first_noise[
        "evaluation_records"
    ]
    assert continued_records[first_count]["evaluation_ordinal"] == (
        first_count + 1
    )
    assert continued_records[first_count]["value_noise"][
        "draw_index_start"
    ] == first_noise["value_noise"]["draw_count"]
    assert continued_noise["value_noise"]["draw_count"] > first_noise[
        "value_noise"
    ]["draw_count"]

    fresh = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=2,
            observation=request.observation,
        ),
    )
    fresh_noise = fresh.scientific_receipts["controller_noise"]
    assert [row.to_dict() for row in continued.run.accepted_trajectory] == [
        row.to_dict() for row in fresh.run.accepted_trajectory
    ]
    assert continued_noise["evaluation_records"] == fresh_noise[
        "evaluation_records"
    ]
    assert continued_noise["compiled_noise_receipts"] == fresh_noise[
        "compiled_noise_receipts"
    ]
    assert continued_noise["final_controller_energy"] == fresh_noise[
        "final_controller_energy"
    ]
    assert continued_noise["final_exact_diagnostic_energy"] == fresh_noise[
        "final_exact_diagnostic_energy"
    ]
    deltas = [
        row["controller_noise"]["runtime_delta"]
        for row in continued.scientific_receipts["accepted_round_receipts"]
    ]
    assert all("runtime_checkpoint" not in row["controller_noise"] for row in (
        continued.scientific_receipts["accepted_round_receipts"]
    ))
    assert [delta["evaluation_count_before"] for delta in deltas] == [
        0,
        first_count,
    ]
    assert sum(
        len(delta["evaluation_records_delta"]) for delta in deltas
    ) == continued_noise["evaluation_count"]
    assert any(len(snapshot) == 2 for snapshot in (
        live_history_controller_noise_schemas
    ))
    assert all(
        schema
        == "paper_i_pure_hubbard_controller_noise_checkpoint_binding_v1"
        for snapshot in live_history_controller_noise_schemas
        for schema in snapshot
    )
    assert any(
        len(snapshot) == 2
        for snapshot in retained_projected_round_controller_noise
    )
    assert all(
        schema
        == "paper_i_pure_hubbard_controller_noise_checkpoint_binding_v1"
        and evaluation_record_count == 0
        for snapshot in retained_projected_round_controller_noise
        for schema, evaluation_record_count in snapshot
    )
