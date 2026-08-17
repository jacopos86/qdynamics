from __future__ import annotations

from pathlib import Path
import inspect

import pytest

from chtc.paper_i_ra_adapt_repair_20260727 import (
    run_local_paper_i_adaptive_append_endpoint_phase0_sw_always_k5_pair_20260816
    as runner,
)


def _summary(*, energy: float, error: float, s_alg: int) -> dict[str, object]:
    return {
        "available_controller_rounds": 5,
        "accepted_error_trace": [
            {
                "controller_round": index,
                "accepted_energy": energy + (5 - index) * 0.01,
                "absolute_energy_error": error + (5 - index) * 0.01,
                "exact_same_cutoff_energy": 0.5,
                "active_ansatz_depth": index,
            }
            for index in range(1, 6)
        ],
        "canonical_all_work": {
            "components": {
                "n_h_outer": 5,
                "n_h_refit": 10,
                "n_grad": s_alg - 22,
                "n_metric": 7,
            },
            "s_alg": s_alg,
        },
    }


def test_exact_two_cell_contract_is_fixed_order_k5_and_not_adopted() -> None:
    assert [cell.mode for cell in runner.CELL_SPECS] == [
        "fixed24_graph_weighted_adaptive_shadow_v1",
        "active_adaptive_graph_weighted_v1",
    ]
    assert len({cell.execution_id for cell in runner.CELL_SPECS}) == 2
    assert len({cell.mode for cell in runner.CELL_SPECS}) == 2
    for cell in runner.CELL_SPECS:
        assert cell.regime_id == "strong_weak_u8"
        assert cell.nph == 3
        assert cell.target_horizon == 5
        assert cell.insertion_policy == "always_commutation_reduced"
        assert cell.fresh_start is True
        assert cell.submission_authorized is False
        assert cell.paper_adoption_authorized is False
        assert cell.paper_evidence_adoption_authorized is False


def test_source_binding_closes_exact_sealed_parent() -> None:
    binding = runner.source_bindings()
    assert binding["source_archive"]["sha256"] == (
        "690d54dbf5bafcaaf974dc11339ed927cb7f5d117265ed51adbb811785740762"
    )
    assert binding["parent_protocol_sha256"] == (
        "37d77f48342cf29f70bcb9710840be0e4a4b7e7d2aac28e8e4dd0cad559064f1"
    )
    assert binding["parent_route_contract_sha256"] == (
        "24d5aed82ee202293187deb5e9745875a5779f8d6bca806536e4a323c7a307a6"
    )


def test_capacity_gate_uses_only_generic_ram_and_disk() -> None:
    blocked = runner.capacity_snapshot(
        available_memory_bytes=runner.MIN_LAUNCH_AVAILABLE_MEMORY_BYTES - 1,
        free_disk_bytes=runner.MIN_LAUNCH_FREE_DISK_BYTES,
    )
    assert blocked["status"] == "waiting_for_launch_capacity"
    assert blocked["launch_ready"] is False
    assert set(blocked) >= {
        "available_memory_bytes",
        "free_disk_bytes",
        "minimum_launch_available_memory_bytes",
        "minimum_launch_free_disk_bytes",
    }
    assert all(
        forbidden not in blocked
        for forbidden in ("processes", "overlaps", "external_jobs", "predecessor")
    )

    ready = runner.capacity_snapshot(
        available_memory_bytes=runner.MIN_LAUNCH_AVAILABLE_MEMORY_BYTES,
        free_disk_bytes=runner.MIN_LAUNCH_FREE_DISK_BYTES,
    )
    assert ready["status"] == "ready_for_adaptive_pair"
    assert ready["launch_ready"] is True


def test_capacity_wait_is_bounded_and_emits_honest_status() -> None:
    states: list[dict[str, object]] = []
    clock_values = iter([0.0, 0.0, 2.0, 6.0])
    with pytest.raises(runner.RunnerError, match="bounded capacity wait timed out"):
        runner.wait_for_launch_capacity(
            maximum_wait_seconds=5.0,
            poll_seconds=0.0,
            clock=lambda: next(clock_values),
            sleeper=lambda _seconds: None,
            memory_supplier=lambda: runner.MIN_LAUNCH_AVAILABLE_MEMORY_BYTES - 1,
            disk_supplier=lambda: runner.MIN_LAUNCH_FREE_DISK_BYTES,
            status_sink=states.append,
        )
    assert states[0]["status"] == "waiting_for_launch_capacity"
    assert states[-1]["status"] == "failed_capacity_wait_timeout"
    assert all(state.get("launch_ready") is False for state in states)


def test_dedicated_campaign_lock_rejects_a_second_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "campaign.lock"
    with runner.campaign_lock(lock_path):
        with pytest.raises(runner.RunnerError, match="already owns its lock"):
            with runner.campaign_lock(lock_path):
                pass


def test_pristine_cell_guard_rejects_partial_or_duplicate_output(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    staging_dir = tmp_path / "staging"
    receipt_path = tmp_path / "receipt.json"
    runner.assert_pristine_cell_paths(run_dir, staging_dir, receipt_path)
    staging_dir.mkdir()
    with pytest.raises(runner.RunnerError, match="not pristine"):
        runner.assert_pristine_cell_paths(run_dir, staging_dir, receipt_path)


def test_comparison_receipt_is_exact_and_uses_fixed_cell_order() -> None:
    summaries = {
        runner.CELL_SPECS[0].mode: _summary(energy=0.51, error=0.01, s_alg=100),
        runner.CELL_SPECS[1].mode: _summary(energy=0.505, error=0.005, s_alg=80),
    }
    receipt = runner.build_terminal_comparison(
        summaries=summaries,
        worker_receipt_sha256_by_mode={
            runner.CELL_SPECS[0].mode: "a" * 64,
            runner.CELL_SPECS[1].mode: "b" * 64,
        },
        guard_receipt_sha256_by_mode={
            runner.CELL_SPECS[0].mode: "c" * 64,
            runner.CELL_SPECS[1].mode: "d" * 64,
        },
        capacity_receipt_sha256="e" * 64,
    )
    assert receipt["status"] == "passed_exact_two_cells_k5"
    assert [row["mode"] for row in receipt["cells"]] == [
        cell.mode for cell in runner.CELL_SPECS
    ]
    assert receipt["comparison"]["adaptive_minus_fixed"][
        "absolute_energy_error"
    ] == pytest.approx(-0.005)
    assert receipt["comparison"]["adaptive_minus_fixed"]["s_alg"] == -20
    assert receipt["submission_authorized"] is False
    assert receipt["paper_adoption_authorized"] is False
    assert receipt["paper_evidence_adoption_authorized"] is False


def test_plan_is_diagnostic_cpu_only_and_never_authorizes_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "source_bindings", lambda: {"source": "bound"})
    monkeypatch.setattr(
        runner,
        "file_binding",
        lambda path: {"path": str(path), "sha256": "f" * 64, "size_bytes": 1},
    )
    plan = runner.build_plan()
    assert plan["run_class"] == "diagnostic"
    assert plan["maximum_concurrency"] == 1
    assert plan["coordination_scope"] == "dedicated_campaign_lock_and_capacity_only"
    assert plan["runtime_environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert plan["execution_authorized"] is False
    assert plan["submission_authorized"] is False
    assert plan["paper_adoption_authorized"] is False
    assert plan["paper_evidence_adoption_authorized"] is False
    assert [row["mode"] for row in plan["cells"]] == [
        cell.mode for cell in runner.CELL_SPECS
    ]


def test_native_semantic_closure_is_explicit_and_execution_stays_disabled() -> None:
    semantics = runner.REQUIRED_NATIVE_SEMANTICS
    assert semantics["phase0_variants"]["gradient_only"] == {
        "population": "same_append_endpoint_generator_population",
        "ranking_signal": "absolute_gradient",
        "structural_proxy_active": False,
        "filesystem_metric_active": False,
        "qiskit_active": False,
    }
    proxy = semantics["phase0_variants"]["proxy_cost"]
    assert proxy["population"] == "same_append_endpoint_generator_population"
    assert proxy["structural_proxy_active"] is True
    assert proxy["filesystem_metric_active"] is False
    assert proxy["qiskit_active"] is False
    assert semantics["compile_scope"] == "phase0_proxy_or_off"
    assert semantics["qiskit_active_phases"] == [
        "phase_i",
        "phase_ii",
        "phase_iii",
    ]
    assert semantics["compile_ansatz_scope"] == (
        "full_base_and_trial_ansatz_at_recorded_insertion_position"
    )
    assert semantics["signed_compile_deltas"] == ["dN2q", "dD2q", "dN1q"]
    assert semantics["signed_delta_transform"] == (
        "zero_centered_signed_arctan_v1"
    )
    assert semantics["negative_cancellation_rewarded"] is True
    assert semantics["selection_factor_active_phases"] == [
        "phase_i",
        "phase_ii",
        "phase_iii",
    ]
    assert semantics["s_alg_includes_compile_work"] is False
    assert runner.EXECUTION_SURFACE_ENABLED is False
    with pytest.raises(runner.RunnerError, match="semantic closure is unresolved"):
        runner.assert_native_route_ready()
    with pytest.raises(runner.RunnerError, match="semantic closure is unresolved"):
        runner.materialize_authority()


def test_scaffold_contains_no_external_job_inspection_or_launch_cli() -> None:
    source = inspect.getsource(runner)
    assert "process_iter" not in source
    assert "run_local_batch.sh" not in source
    assert "REMOTE_JOB_ID" not in source
    assert 'mode.add_argument("--run"' not in source
    assert "temporary monkeypatch overlay is intentionally not an execution path" in source
