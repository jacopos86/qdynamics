"""Route-object integration of the exchange selector."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    SupportPatchControllerConfig,
    _active_prune_atoms,
    _PruneControllerRuntimeState,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_integration import (
    build_population_ranker,
    select_generalized_exchange_patch,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_structural import (
    StructuralCandidate,
    StructuralScoreWeights,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import solve_fixed_mclachlan_step
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import (
    evaluate_mclachlan_geometry,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.state import state_from_scaffold_runtime_input
from types import SimpleNamespace

from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(*components: tuple[str, float], nq: int = 2) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for word, coeff in components:
        poly.add_term(PauliTerm(int(nq), ps=str(word), pc=float(coeff)))
    poly._reduce()
    return poly


X0 = AnsatzTerm(label="sx0", polynomial=_poly(("ex", 1.0)))
Z0 = AnsatzTerm(label="sz0", polynomial=_poly(("ez", 1.0)))
CA = AnsatzTerm(label="candidate_ca", polynomial=_poly(("xe", 0.7)))
CB = AnsatzTerm(label="candidate_cb", polynomial=_poly(("ye", 0.4)))
HAM = TimeDependentHamiltonian(static_poly=_poly(("ez", 2.0), ("xx", 0.6)))
POLICY = McLachlanInversePolicy(pinv_rcond=1.0e-10, ridge_lambda=1.0e-7)


def _state(theta=(0.05, -0.04)):
    selected = (X0, Z0)
    layout = build_parameter_layout(selected)
    executor = CompiledAnsatzExecutor(
        tuple(selected),
        parameterization_layout=layout,
        parameterization_mode="per_pauli_term",
    )
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0
    psi_initial = executor.prepare_state(np.asarray(theta, dtype=float), psi_ref)
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(
            family_key="toy", hamiltonian=_poly(("ez", 2.0), ("xx", 0.6))
        ),
        psi_ref=psi_ref,
        psi_initial=np.asarray(psi_initial, dtype=complex),
        base_layout=layout,
        theta_runtime=np.asarray(theta, dtype=float),
        theta_logical=np.zeros(int(layout.logical_parameter_count), dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=(CA, CB),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool", pool_key="toy_pool", completeness="complete"
        ),
        provenance={"artifact_json": "toy.json"},
    )
    return state_from_scaffold_runtime_input(runtime_input)


def _run(config: SupportPatchControllerConfig, theta=(0.05, -0.04)):
    state = _state(theta)
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )
    step = solve_fixed_mclachlan_step(evaluation.geometry, inverse_policy=POLICY)
    return select_generalized_exchange_patch(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        base_evaluation=evaluation,
        base_step=step,
        inverse_policy=POLICY,
        support_config=config,
        runtime_state=_PruneControllerRuntimeState(),
        time_index=3,
    )


def _config(**overrides):
    base = dict(
        residual_ratio_threshold=0.0,
        prune_ray_distance_tol=1.0,
        prune_patch_smoothness_eta_max=1.0e6,
        min_runtime_parameter_count=1,
        max_insertion_batch_size=1,
    )
    base.update(overrides)
    return SupportPatchControllerConfig(**base)


def test_selector_commits_from_real_route_objects_and_reports_payload() -> None:
    selection, payload = _run(_config())
    assert selection.kind in {"insert", "delete", "exchange"}
    assert payload["selection_policy"] == "paper_ii_generalized_exchange_v2"
    assert payload["kind"] == selection.kind
    assert payload["attempt_count"] == len(selection.attempts)
    assert payload["cost_normalization"] == "per_candidate_raw_v1"
    committed = payload["committed"]
    assert committed["score"] == pytest.approx(selection.committed.score)
    # Certified route objects exist and the patched state differs structurally.
    assert selection.certification.state is not None
    if selection.kind == "insert":
        assert (
            selection.certification.state.runtime_parameter_count
            > 2
        )
    # Inserted coordinate labels are the occurrence labels (time index tagged).
    if selection.committed.inserted_selection:
        labels = selection.certification.state.runtime_coordinate_labels
        assert any("::insr3c" in lab for lab in labels)
    work = payload["estimator_work"]
    assert work["schema"] == "paper_ii_generalized_exchange_estimator_work_v1"
    assert work["selector_pass_count"] == len(work["selector_passes"])
    assert work["selector_pass_count"] >= 1
    for selector_pass in work["selector_passes"]:
        p = int(selector_pass["pre_runtime_parameter_count"])
        columns = int(selector_pass["positioned_candidate_column_count"])
        assert selector_pass["logical_shared_cache_entry_count"] == (
            p * columns + columns * (columns + 3) // 2
        )
        assert selector_pass["complete"] is True
    assert work["logical_shared_cache_entry_count"] == sum(
        int(row["logical_shared_cache_entry_count"])
        for row in work["selector_passes"]
    )
    assert work["certification_attempt_count"] == len(
        selection.certification_work
    )
    assert work["certification_attempts"] == [
        dict(row) for row in selection.certification_work
    ]
    assert sum(work["attempt_reason_counts"].values()) == work[
        "attempt_record_count"
    ]
    assert work["attempt_record_count"] == len(selection.attempts)
    assert work["logical_total_selector_estimator_entries"] == (
        work["logical_shared_cache_entry_count"]
        + work["logical_finalist_geometry_entry_count"]
        + work["logical_refit_entry_count"]
    )
    assert work["complete"] is True


def test_work_guard_and_frontier_telemetry_flow_through_payload() -> None:
    selection, payload = _run(
        _config(
            prune_ray_distance_tol=1.0e-15,
            prune_patch_smoothness_eta_max=1.0e-15,
            structural_score_floor=1.0e30,
        )
    )
    assert selection.kind == "stay"
    assert payload["kind"] == "stay"
    assert "work_guard" in payload
    assert payload["work_guard"]["rejected_family"] is None
    assert payload["attempt_count"] == 0


def test_escalation_respects_residual_threshold_predicate() -> None:
    # A score floor removes every d0 candidate; with the residual threshold
    # far above the toy residual, escalation stops before any rung opens.
    selection, payload = _run(
        _config(
            residual_ratio_threshold=1.0e9,
            structural_score_floor=1.0e30,
        )
    )
    assert selection.kind == "stay"
    assert selection.stop_reason == "escalation_predicate_false"
    assert selection.attempts == ()


def test_internal_insert_fallback_preserves_both_selector_work_passes(
    monkeypatch,
) -> None:
    from pipelines.time_dynamics.ap_mclachlan import exchange_integration as module

    dispositions = iter(("retry_insert_face", "accept"))

    def force_one_fallback(**_kwargs):
        return next(dispositions)

    monkeypatch.setattr(
        module.GeneralizedExchange,
        "assess_realized_candidate",
        staticmethod(force_one_fallback),
    )
    selection, payload = _run(_config())

    assert payload["deletion_fallback_used"] is True
    work = payload["estimator_work"]
    assert work["selector_pass_count"] == 2
    assert [row["pass_index"] for row in work["selector_passes"]] == [0, 1]
    assert work["selector_passes"][0]["allowed_deletion_atom_count"] > 0
    assert work["selector_passes"][1]["allowed_deletion_atom_count"] == 0
    assert work["logical_shared_cache_entry_count"] == sum(
        int(row["logical_shared_cache_entry_count"])
        for row in work["selector_passes"]
    )
    assert work["certification_attempt_count"] == len(
        selection.certification_work
    )


def test_min_surviving_support_bounds_deletions() -> None:
    selection, payload = _run(
        _config(
            min_runtime_parameter_count=2,
            prune_ray_distance_tol=1.0e-15,
            prune_patch_smoothness_eta_max=1.0e-15,
        )
    )
    # With 2 active coordinates and min survivors 2, no deletion is feasible:
    # every attempt is insert-kind only.
    kinds = {a.kind for a in selection.attempts}
    assert "delete" not in kinds and "exchange" not in kinds


def test_config_lambdas_map_onto_structural_weights() -> None:
    from pipelines.time_dynamics.ap_mclachlan.exchange_integration import (
        build_selector_inputs,
    )

    state = _state()
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )
    inputs = build_selector_inputs(
        state=state,
        evaluation=evaluation,
        support_config=_config(
            prune_condition_lambda_kappa_rel=2.5,
            prune_condition_lambda_kappa_dam=1.5,
            prune_history_lambda=0.75,
        ),
        runtime_state=_PruneControllerRuntimeState(),
        theta_runtime=state.theta_runtime,
        time_index=0,
        deletable_atoms=_active_prune_atoms(
            state,
            theta_runtime=state.theta_runtime,
            support_config=_config(
                prune_condition_lambda_kappa_rel=2.5,
                prune_condition_lambda_kappa_dam=1.5,
                prune_history_lambda=0.75,
            ),
            runtime_state=_PruneControllerRuntimeState(),
            time_index=0,
        ),
    )
    weights = inputs["structural_kwargs"]["weights"]
    assert weights.lambda_cond_relief == 2.5
    assert weights.lambda_cond_damage == 1.5
    assert weights.lambda_hist == 0.75


def test_history_prior_reads_windowed_mean_from_runtime_state() -> None:
    from pipelines.time_dynamics.ap_mclachlan.exchange_integration import (
        build_selector_inputs,
    )

    state = _state()
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )
    labels = tuple(state.runtime_coordinate_labels)
    runtime_state = _PruneControllerRuntimeState()
    # Three records for coordinate 0; window 2 keeps only the last two.
    runtime_state.loss_history[labels[0]] = [(0, 100.0), (1, 4.0), (2, 8.0)]
    inputs = build_selector_inputs(
        state=state,
        evaluation=evaluation,
        support_config=_config(prune_history_window=2),
        runtime_state=runtime_state,
        theta_runtime=state.theta_runtime,
        time_index=3,
        deletable_atoms=_active_prune_atoms(
            state,
            theta_runtime=state.theta_runtime,
            support_config=_config(prune_history_window=2),
            runtime_state=runtime_state,
            time_index=3,
        ),
    )
    prior = inputs["structural_kwargs"]["deletion_history_loss"]
    assert prior(()) == 0.0
    assert prior((0,)) == pytest.approx(6.0)
    # Coordinate 1 has no history: it contributes zero to the mean.
    assert prior((0, 1)) == pytest.approx(3.0)


def test_recorder_persists_attempted_deletion_losses_with_window() -> None:
    # Zero angles and permissive measurement-free permission make deletion
    # branches reach the deliberately impossible conditioning gate below.
    state = _state(theta=(0.0, 0.0))
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )
    step = solve_fixed_mclachlan_step(evaluation.geometry, inverse_policy=POLICY)
    runtime_state = _PruneControllerRuntimeState()
    config = _config(
        prune_history_window=2,
        # Impossible gates (conditioning included, so the insert-only d0
        # level cannot commit): every ranked candidate at every level is
        # attempted, fails, and the deletion-bearing ones get recorded.
        prune_ray_distance_tol=1.0,
        prune_patch_smoothness_eta_max=1.0e6,
        append_schur_max_condition_number=1.0e-30,
    )

    def run_once(time_index):
        return select_generalized_exchange_patch(
            state=state,
            hamiltonian=HAM,
            theta_runtime=state.theta_runtime,
            time=0.0,
            base_evaluation=evaluation,
            base_step=step,
            inverse_policy=POLICY,
            support_config=config,
            runtime_state=runtime_state,
            time_index=time_index,
        )

    selection, _payload = run_once(3)
    deletion_attempts = [
        a for a in selection.attempts if a.removed_runtime_indices
    ]
    assert deletion_attempts, "fixture must attempt deletion-bearing patches"
    assert runtime_state.loss_history
    labels = tuple(state.runtime_coordinate_labels)
    for attempt in deletion_attempts:
        assert attempt.deletion_loss is not None
        for index in attempt.removed_runtime_indices:
            records = runtime_state.loss_history[labels[int(index)]]
            assert all(t == 3 for t, _loss in records)

    # A second checkpoint appends and the window trims to the newest two.
    run_once(4)
    for records in runtime_state.loss_history.values():
        assert 1 <= len(records) <= 2
    assert any(
        t == 4 for records in runtime_state.loss_history.values() for t, _ in records
    )


def test_certification_refit_flag_runs_through_route_adapter() -> None:
    selection, payload = _run(
        _config(
            certification_refit_enabled=True,
            certification_refit_trust_radius=0.3,
            certification_refit_max_iterations=10,
        )
    )
    assert selection.kind in {"insert", "delete", "exchange", "stay"}
    if selection.committed is not None:
        assert selection.certification.certified
    work = payload["estimator_work"]
    assert work["refit_objective_evaluation_attempt_count"] == work[
        "refit_objective_evaluation_count"
    ]
    assert work["logical_refit_entry_count"] >= work[
        "refit_objective_evaluation_count"
    ]
    assert work["complete"] is True


def test_attempt_budget_bounds_certifications_per_level() -> None:
    # Impossible gates (conditioning included) force every attempt to fail;
    # without a budget the selector would attempt every ranked candidate.
    unbounded, _ = _run(
        _config(
            prune_ray_distance_tol=1.0,
            prune_patch_smoothness_eta_max=1.0e6,
            append_schur_max_condition_number=1.0e-30,
        ),
        theta=(0.0, 0.0),
    )
    assert unbounded.kind == "stay"
    assert len(unbounded.attempts) > 2

    bounded, payload = _run(
        _config(
            prune_ray_distance_tol=1.0,
            prune_patch_smoothness_eta_max=1.0e6,
            append_schur_max_condition_number=1.0e-30,
            max_certification_attempts_per_level=1,
        ),
        theta=(0.0, 0.0),
    )
    assert bounded.kind == "stay"
    assert bounded.stop_reason == "attempt_budget_exhausted"
    # One attempt per acquired level, never the unbounded grind.
    assert 0 < len(bounded.attempts) < len(unbounded.attempts)
    assert payload["stop_reason"] == "attempt_budget_exhausted"


def test_prune_only_mode_empties_insertion_pool_but_keeps_deletions() -> None:
    state = _state(theta=(0.0, 0.0))
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )
    step = solve_fixed_mclachlan_step(evaluation.geometry, inverse_policy=POLICY)
    selection, payload = select_generalized_exchange_patch(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        base_evaluation=evaluation,
        base_step=step,
        inverse_policy=POLICY,
        support_config=_config(
            prune_ray_distance_tol=1.0,
            prune_patch_smoothness_eta_max=1.0e6,
            insertion_gate_mode="mclachlan_l2",
            insertion_l2_cut=1.0e9,
        ),
        runtime_state=_PruneControllerRuntimeState(),
        time_index=3,
    )
    assert payload["insertions_enabled"] is False
    assert payload["candidate_pool_deduplicated"] == 0
    # Measurement-free pure deletions are still enumerated and attempted.
    kinds = {a.kind for a in selection.attempts}
    assert kinds <= {"delete"} and "delete" in kinds


def test_measurement_free_permission_blocks_deletion_before_certification() -> None:
    selection, payload = _run(
        _config(
            insertion_gate_mode="mclachlan_l2",
            insertion_l2_cut=1.0e9,
            prune_ray_distance_tol=1.0e-3,
            prune_patch_smoothness_eta_max=1.0e6,
        ),
        theta=(0.05, -0.04),
    )

    assert selection.attempts == ()
    permission = payload["deletion_permission"]
    assert permission["evaluated_deletion_set_count"] > 0
    assert permission["permitted_deletion_set_count"] == 0
    assert permission["rejection_reason_counts"] == {
        "angle_ray_upper_bound_above_max": permission[
            "evaluated_deletion_set_count"
        ]
    }


def test_generalized_exchange_payload_identical_and_worker_telemetry_recorded() -> None:
    import json as json_module

    serial_selection, serial_payload = _run(_config())
    parallel_selection, parallel_payload = _run(
        _config(support_patch_scoring_workers=4)
    )

    def stripped(payload):
        out = dict(payload)
        out.pop("structural_scoring_parallelism")
        return json_module.dumps(out, sort_keys=True, default=str)

    # Every scientific field — committed patch, attempts, scores, work guard,
    # q_base, permission summary — is identical; only the parallelism
    # telemetry block may differ.
    assert stripped(parallel_payload) == stripped(serial_payload)
    assert parallel_selection.kind == serial_selection.kind

    serial_tel = serial_payload["structural_scoring_parallelism"]
    assert serial_tel["schema"] == "paper_ii_exchange_scoring_parallelism_v1"
    assert serial_tel["requested_workers"] == 1
    assert serial_tel["execution_mode"] == "serial"
    parallel_tel = parallel_payload["structural_scoring_parallelism"]
    assert parallel_tel["requested_workers"] == 4
    assert parallel_tel["execution_mode"] == "speculative_memo_threads"
    for tel in (serial_tel, parallel_tel):
        assert set(tel["blas_thread_env"]) == {
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        }
        assert (
            tel["worker_clamp_policy"]
            == "min_requested_and_family_missing_solve_count"
        )


def test_legacy_proxy_fixed_bundle_normalizes_cost_over_the_population() -> None:
    # Same physics as the historical route, but C_I / C_D come from the
    # competing population instead of a singleton (which made them 1.0).
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
        EXCHANGE_ROUTE_BUNDLE_LEGACY_PROXY_FIXED,
    )

    historical_selection, historical_payload = _run(_config())
    selection, payload = _run(
        _config(exchange_route_bundle=EXCHANGE_ROUTE_BUNDLE_LEGACY_PROXY_FIXED)
    )
    route = payload["selector_route"]
    assert route["bundle"]["bundle_id"] == EXCHANGE_ROUTE_BUNDLE_LEGACY_PROXY_FIXED
    assert route["bundle"]["kernel_id"] == "legacy_utility_debt_ordering_v2"
    assert payload["cost_normalization"] == "paper_i_proxy_population_v1"
    assert route["regime"] in {"large_miss", "maintenance"}
    assert route["levels"], "population levels must be recorded"
    level = route["levels"][0]
    assert level["population_size"] > 0
    assert level["ranked_count"] > 0
    # The historical path never records these; the bundle does.
    assert historical_payload["selector_route"]["bundle"]["bundle_id"] == (
        "paper_ii_generalized_exchange_v2"
    )
    assert historical_payload["selector_route"]["levels"] == []
    # Attempts carry the population denominator that ranked them.
    assert all(a["cost_factor_k"] is not None for a in payload["attempts"])
    assert all(a["cost_factor_k"] >= 1.0 for a in payload["attempts"])
    assert selection.kind in {"insert", "delete", "exchange", "stay"}


def test_population_bundle_records_gamma_for_compiled_routes_only() -> None:
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
        EXCHANGE_ROUTE_BUNDLE_LEGACY_PROXY_FIXED,
    )

    _selection, payload = _run(
        _config(exchange_route_bundle=EXCHANGE_ROUTE_BUNDLE_LEGACY_PROXY_FIXED)
    )
    assert payload["selector_route"]["gamma_rcond"] is None
    assert payload["selector_route"]["compile"] is None
    assert payload["selector_route"]["maintenance_loss_tolerance"] == _config().prune_patch_smoothness_eta_max


def _route_payload(bundle_id, **overrides):
    return _run(_config(exchange_route_bundle=bundle_id, **overrides))


def test_q_and_gamma_compiled_bundles_share_population_and_cost_map() -> None:
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
        EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED,
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
    )

    q_selection, q_payload = _route_payload(EXCHANGE_ROUTE_BUNDLE_Q_COMPILED)
    g_selection, g_payload = _route_payload(EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED)
    for payload, bundle in ((q_payload, EXCHANGE_ROUTE_BUNDLE_Q_COMPILED),
                            (g_payload, EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED)):
        route = payload["selector_route"]
        assert route["bundle"]["bundle_id"] == bundle
        assert route["bundle"]["kernel_id"] == "shifted_paper_i_v1"
        assert route["bundle"]["compiler"]["backend_name"] == "FakeMarrakesh"
        assert route["gamma_rcond"] is not None
        compile_block = route["compile"]
        assert compile_block["placement_rule"].endswith("placement_v1")
        assert compile_block["compile_calls"] >= 1
        assert compile_block["backend"]["backend_name"] == "FakeMarrakesh"
        assert route["levels"] and route["levels"][0]["population_size"] > 0
        assert payload["cost_normalization"] == "compiled_marrakesh_population_v1"
        for attempt in payload["attempts"]:
            assert attempt["cost_factor_k"] is not None and attempt["cost_factor_k"] > 0.0
            assert attempt["route_score"] is not None
    # Identical prefilter population and K map: only the ordinate differs.
    q_level, g_level = q_payload["selector_route"]["levels"][0], g_payload["selector_route"]["levels"][0]
    assert q_level["population_sha256"] == g_level["population_sha256"]
    assert q_level["medians"] == g_level["medians"] and q_level["mads"] == g_level["mads"]
    q_k = {tuple(map(str, row["patch"])): row["K"] for row in q_level["top"]}
    g_k = {tuple(map(str, row["patch"])): row["K"] for row in g_level["top"]}
    shared = set(q_k) & set(g_k)
    assert shared and all(q_k[key] == g_k[key] for key in shared)
    assert q_payload["selector_route"]["bundle"]["objective_id"] != (
        g_payload["selector_route"]["bundle"]["objective_id"]
    )


def test_q_and_gamma_share_the_batch_two_frontier_and_cost_receipts() -> None:
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
        EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED,
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
    )
    from pipelines.time_dynamics.ap_mclachlan.exchange_integration import (
        build_selector_inputs,
    )
    from pipelines.time_dynamics.ap_mclachlan.exchange_structural import (
        enumerate_structural_candidates,
    )

    state = _state()
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )
    results = []
    for bundle in (
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
        EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED,
    ):
        config = _config(
            exchange_route_bundle=bundle,
            max_insertion_batch_size=2,
            interaction_frontier_widths=(2,),
            max_joint_patch_evaluations=50000,
        )
        runtime_state = _PruneControllerRuntimeState()
        inputs = build_selector_inputs(
            state=state,
            evaluation=evaluation,
            support_config=config,
            runtime_state=runtime_state,
            theta_runtime=state.theta_runtime,
            time_index=3,
            deletable_atoms=_active_prune_atoms(
                state,
                theta_runtime=state.theta_runtime,
                support_config=config,
                runtime_state=runtime_state,
                time_index=3,
            ),
            insertions_enabled=True,
            debt_ranking=True,
            accuracy_debt=True,
        )
        inputs["structural_kwargs"]["inverse_policy"] = POLICY
        try:
            results.append(enumerate_structural_candidates(**inputs["structural_kwargs"]))
        finally:
            inputs["compiled_cost_runtime"].close()
    q_result, g_result = results
    assert q_result.frontiers_used == g_result.frontiers_used == 1
    assert q_result.eligible_universe == g_result.eligible_universe
    q_map = {c.order_key: c.cost_marginal for c in q_result.candidates}
    g_map = {c.order_key: c.cost_marginal for c in g_result.candidates}
    assert q_map == g_map
    assert any(candidate.plan is not None for candidate in q_result.candidates)


@pytest.mark.parametrize(
    "scoring_workers,compile_workers",
    [(4, None), (1, 8), (4, 8)],
    ids=["scoring4", "compile8", "scoring4_compile8"],
)
def test_compiled_bundle_is_bitwise_identical_across_scoring_workers(
    scoring_workers, compile_workers
) -> None:
    import json as json_module
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
    )

    serial_selection, serial_payload = _route_payload(EXCHANGE_ROUTE_BUNDLE_Q_COMPILED)
    parallel_selection, parallel_payload = _route_payload(
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
        support_patch_scoring_workers=scoring_workers,
        compile_workers=compile_workers,
    )
    expected_pool = scoring_workers if compile_workers is None else compile_workers
    assert parallel_payload["selector_route"]["compile"]["compile_workers"] == expected_pool

    def stripped(payload):
        out = dict(payload)
        out.pop("structural_scoring_parallelism")
        route = dict(out["selector_route"])
        compile_block = dict(route["compile"])
        # Execution-resource telemetry differs by construction; every
        # scientific field must not.
        for key in (
            "compile_seconds",
            "warm_seconds",
            "warm_primed_words",
            "compile_workers",
            "compile_execution_mode",
            "worker_blas_env_pinned",
            "cache_hits",
        ):
            compile_block.pop(key)
        route["compile"] = compile_block
        # After a warm every candidate is a cache hit inside the enumeration,
        # so this counter is 0; the real compile total is compared unstripped.
        route.pop("compile_calls_in_enumeration")
        out["selector_route"] = route
        return json_module.dumps(out, sort_keys=True, default=str)

    assert stripped(parallel_payload) == stripped(serial_payload)
    assert parallel_selection.kind == serial_selection.kind


def test_gamma_rejection_preserves_serial_parallel_compile_parity(
    monkeypatch,
) -> None:
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
    )
    from pipelines.time_dynamics.ap_mclachlan.objective_ordinates import (
        GammaMonotonicityError,
    )

    def reject_pure_insertion(**_kwargs):
        raise GammaMonotonicityError("synthetic rank transition")

    monkeypatch.setattr(
        "pipelines.time_dynamics.ap_mclachlan.exchange_structural."
        "check_gamma_nested_monotonicity",
        reject_pure_insertion,
    )
    serial_selection, serial = _route_payload(
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
        support_patch_scoring_workers=1,
        insertion_gate_mode="mclachlan_l2",
        insertion_l2_cut=0.0,
    )
    parallel_selection, parallel = _route_payload(
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
        support_patch_scoring_workers=4,
        insertion_gate_mode="mclachlan_l2",
        insertion_l2_cut=0.0,
    )
    serial_route = serial["selector_route"]
    parallel_route = parallel["selector_route"]
    assert serial_route["compile"]["compile_calls"] == parallel_route["compile"][
        "compile_calls"
    ]
    assert [level["population_sha256"] for level in serial_route["levels"]] == [
        level["population_sha256"] for level in parallel_route["levels"]
    ]
    assert [
        level["decision_receipt_sha256"] for level in serial_route["levels"]
    ] == [level["decision_receipt_sha256"] for level in parallel_route["levels"]]
    serial_rejections = [
        row
        for level in serial_route["levels"]
        for row in level["ordinate_rejections"]
    ]
    parallel_rejections = [
        row
        for level in parallel_route["levels"]
        for row in level["ordinate_rejections"]
    ]
    assert serial_rejections
    assert serial_rejections == parallel_rejections
    assert serial_selection.kind == parallel_selection.kind


def test_compiled_bundle_uses_placement_cut_when_it_materializes() -> None:
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
    )

    selection, payload = _route_payload(EXCHANGE_ROUTE_BUNDLE_Q_COMPILED)
    if selection.committed is None or not selection.committed.inserted_selection:
        return
    placement = dict(selection.committed.placement_cuts or ())
    assert placement, "compiled bundle must record a placement cut per inserted atom"
    labels = selection.certification.state.runtime_coordinate_labels
    for atom_id, cut in placement.items():
        assert any(f"c{int(cut)}o" in lab for lab in labels)


def test_compiled_bundle_pool_warm_records_its_mode_and_primes_words() -> None:
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
    )

    _sel1, serial = _route_payload(EXCHANGE_ROUTE_BUNDLE_Q_COMPILED)
    _sel4, parallel = _route_payload(
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED, support_patch_scoring_workers=2
    )
    s, p = serial["selector_route"]["compile"], parallel["selector_route"]["compile"]
    assert s["compile_execution_mode"] == "spawn_pool_serial" and s["warm_primed_words"] == 0
    assert p["compile_execution_mode"] == "spawn_pool_warm_then_serial_replay"
    assert p["compile_workers"] == 2
    assert p["warm_primed_words"] > 0
    # Every word the serial path compiled on demand was compiled by the pool
    # instead; the totals agree and so does every ranked value.
    assert p["compile_calls"] == s["compile_calls"]
    assert serial["selector_route"]["levels"][0]["top"] == parallel["selector_route"]["levels"][0]["top"]
    assert serial["attempts"] == parallel["attempts"]


def _population_candidate(
    *,
    atom: str | None = None,
    removed: tuple[int, ...] = (),
    delta: float,
    gamma: float | None,
    marginal: tuple[int, int, int] | None,
    cost_error: str | None = None,
    ordinate_error: str | None = None,
) -> StructuralCandidate:
    selection = () if atom is None else ((str(atom), 0),)
    return StructuralCandidate(
        removed_runtime_indices=removed,
        inserted_selection=selection,
        plan=None,
        family="test",
        q=float(delta),
        insertion_gain=max(0.0, float(delta)),
        deletion_loss=max(0.0, -float(delta)),
        delta=float(delta),
        insertion_utility=0.0,
        deletion_utility=0.0,
        score=float(delta),
        m_gamma=None if gamma is None else float(gamma),
        cost_marginal=marginal,
        cost_error=cost_error,
        ordinate_error=ordinate_error,
    )


def test_population_is_closed_by_l2_regime_before_cost_normalization() -> None:
    from pipelines.time_dynamics.ap_mclachlan.append_cost import AppendCostSettings
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
        APGeneralizedExchangeConfig,
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
    )

    route = APGeneralizedExchangeConfig.from_route_config(
        _config(exchange_route_bundle=EXCHANGE_ROUTE_BUNDLE_Q_COMPILED)
    ).route
    delete = _population_candidate(
        removed=(0,), delta=100.0, gamma=100.0, marginal=(-10, -10, -10)
    )
    insert = _population_candidate(
        atom="a", delta=1.0, gamma=1.0, marginal=(1, 1, 1)
    )
    exchange = _population_candidate(
        atom="b", removed=(0,), delta=2.0, gamma=2.0, marginal=(2, 2, 2)
    )
    common = dict(
        bundle=route.bundle,
        route=route,
        weights=StructuralScoreWeights(),
        score_floor=0.0,
        cost_settings=AppendCostSettings(),
        population_context={"checkpoint": 3},
    )
    miss = build_population_ranker(accuracy_debt=True, **common)((delete, insert, exchange))
    assert miss.telemetry["population_size"] == 2
    assert {candidate.kind for candidate in miss.ordered} == {"insert", "exchange"}
    assert delete.order_key not in miss.rank_info

    maintenance = build_population_ranker(accuracy_debt=False, **common)(
        (delete, insert, exchange)
    )
    assert maintenance.telemetry["population_size"] == 1
    assert {candidate.kind for candidate in maintenance.ordered} <= {"delete"}
    assert set(maintenance.rank_info) <= {delete.order_key}


def test_full_population_receipt_exceeds_old_top24_and_recomputes_hash() -> None:
    import hashlib
    import json

    from pipelines.time_dynamics.ap_mclachlan.append_cost import AppendCostSettings
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
        APGeneralizedExchangeConfig,
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
    )

    route = APGeneralizedExchangeConfig.from_route_config(
        _config(exchange_route_bundle=EXCHANGE_ROUTE_BUNDLE_Q_COMPILED)
    ).route
    candidates = tuple(
        _population_candidate(
            atom=f"a{index}",
            delta=float(index + 1),
            gamma=float(index + 2),
            marginal=(index, 2 * index, 3 * index),
        )
        for index in range(30)
    )
    failed = _population_candidate(
        atom="failed",
        delta=999.0,
        gamma=999.0,
        marginal=None,
        cost_error="CompileTimeoutError:timed out",
    )
    level = build_population_ranker(
        bundle=route.bundle,
        route=route,
        accuracy_debt=True,
        weights=StructuralScoreWeights(),
        score_floor=0.0,
        cost_settings=AppendCostSettings(),
        population_context={"checkpoint": 7, "base_word_sha256": "abc"},
    )((*candidates, failed))
    telemetry = level.telemetry
    receipt = telemetry["population_receipt"]
    assert len(receipt["normalization"]["records"]) == 30
    assert len(telemetry["ranked"]) == 30 and len(telemetry["top"]) == 24
    observed = hashlib.sha256(
        json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    assert observed == telemetry["cost_population_sha256"]
    assert telemetry["mad_scales"] == telemetry["scales"]
    assert telemetry["compiler_rejections"] == [
        {
            "patch": [list(failed.order_key[0]), [list(x) for x in failed.order_key[1]]],
            "reason": "CompileTimeoutError:timed out",
        }
    ]


def test_gamma_rejection_is_shared_by_q_and_gamma_compiled_populations() -> None:
    from pipelines.time_dynamics.ap_mclachlan.append_cost import AppendCostSettings
    from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
        APGeneralizedExchangeConfig,
        EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED,
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
    )

    valid = (
        _population_candidate(
            atom="a", delta=1.0, gamma=2.0, marginal=(1, 2, 3)
        ),
        _population_candidate(
            atom="b", delta=2.0, gamma=1.0, marginal=(2, 3, 4)
        ),
    )
    rejected = _population_candidate(
        atom="bad",
        delta=100.0,
        gamma=100.0,
        marginal=(9, 9, 9),
        ordinate_error="GammaMonotonicityError:synthetic rank transition",
    )
    levels = []
    for bundle in (
        EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
        EXCHANGE_ROUTE_BUNDLE_GAMMA_COMPILED,
    ):
        route = APGeneralizedExchangeConfig.from_route_config(
            _config(exchange_route_bundle=bundle)
        ).route
        levels.append(
            build_population_ranker(
                bundle=route.bundle,
                route=route,
                accuracy_debt=True,
                weights=StructuralScoreWeights(),
                score_floor=0.0,
                cost_settings=AppendCostSettings(),
                population_context={"checkpoint": 2},
            )((*valid, rejected))
        )

    q_level, gamma_level = levels
    assert q_level.telemetry["population_size"] == 2
    assert gamma_level.telemetry["population_size"] == 2
    assert q_level.telemetry["population_sha256"] == gamma_level.telemetry["population_sha256"]
    assert q_level.telemetry["cost_population_sha256"] == gamma_level.telemetry[
        "cost_population_sha256"
    ]
    assert rejected.order_key not in q_level.rank_info
    assert rejected.order_key not in gamma_level.rank_info
    expected_rejection = [
        {
            "patch": [
                list(rejected.order_key[0]),
                [list(x) for x in rejected.order_key[1]],
            ],
            "reason": "GammaMonotonicityError:synthetic rank transition",
        }
    ]
    assert q_level.telemetry["ordinate_rejections"] == expected_rejection
    assert gamma_level.telemetry["ordinate_rejections"] == expected_rejection
