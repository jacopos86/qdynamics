from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.reporting.build_paper_i_geo_scaling_evidence import (
    EXPECTED_SCALING_CASE_IDS,
    ordered_scaling_inventory_rows,
    prefix_query_ledger,
    reconstruct_structural_prefix,
    select_first_plateau,
    trajectory_points,
    write_appendix_fragment,
)


def _history_row(
    *,
    position: int,
    error: float,
    energy_before: float,
    energy_after: float,
    depth: int,
    labels: list[str],
    skipped: bool = False,
    nfev: int = 2,
) -> dict[str, object]:
    return {
        "history_position": position,
        "abs_delta_e_same_cutoff_after": error,
        "energy_before": energy_before,
        "energy_after": energy_after,
        "depth_after": depth,
        "selected_batch_labels": labels,
        "appended_operator_count": 0 if skipped else len(labels),
        "geo_immediate_repeat_skipped": skipped,
        "selected_insertion_position": None,
        "outer_hamiltonian_eval_count": 1,
        "optimizer_nfev": nfev,
        "selector_gradient_probe_count": 3,
        "qngd_gradient_operator_probe_count_total": 0,
        "selector_metric_probe_count": 6,
        "qngd_metric_operator_probe_count_total": 0,
        "N_other_quantum": 0,
    }


def test_ordered_scaling_rows_are_exact_not_cartesian() -> None:
    inventory = {
        "rows": [
            {"case_id": case_id, "paper_placement": "appendix_scaling_results"}
            for case_id in EXPECTED_SCALING_CASE_IDS
        ]
        + [{"case_id": "hh_L2_main", "paper_placement": "main_results_hubbard_holstein_L2"}]
    }
    rows = ordered_scaling_inventory_rows(inventory)
    assert tuple(row["case_id"] for row in rows) == EXPECTED_SCALING_CASE_IDS
    assert len(rows) == 34

    inventory["rows"][0], inventory["rows"][1] = inventory["rows"][1], inventory["rows"][0]
    with pytest.raises(ValueError, match="exact ordered 34-case contract"):
        ordered_scaling_inventory_rows(inventory)


def test_first_plateau_is_first_prefix_within_ten_percent_of_best() -> None:
    history = [
        _history_row(position=0, error=5.0, energy_before=4.0, energy_after=5.0, depth=1, labels=["A"]),
        _history_row(position=1, error=1.05, energy_before=5.0, energy_after=1.05, depth=2, labels=["B"]),
        _history_row(position=2, error=1.0, energy_before=1.05, energy_after=1.0, depth=3, labels=["C"]),
    ]
    result = {"same_cutoff_exact_gs_energy": 0.0, "adapt_history": history}
    selected = select_first_plateau(result, horizon=3)
    assert selected.best_observed_error == 1.0
    assert selected.threshold == pytest.approx(1.1)
    assert selected.history_position == 1
    assert selected.k_pl == 2
    assert selected.logical_depth == 2
    assert selected.error_raw == 1.05


def test_trajectory_includes_iteration_zero_and_floors_only_plotted_zero() -> None:
    history = [
        _history_row(position=0, error=0.0, energy_before=2.0, energy_after=0.0, depth=1, labels=["A"])
    ]
    points = trajectory_points({"same_cutoff_exact_gs_energy": 0.0, "adapt_history": history})
    assert [(point.k, point.error_raw) for point in points] == [(0, 2.0), (1, 0.0)]
    assert points[1].error_plotted == 1.0e-16


def test_prefix_query_ledger_sums_native_geo_components() -> None:
    history = [
        _history_row(position=0, error=2.0, energy_before=3.0, energy_after=2.0, depth=1, labels=["A"], nfev=4),
        _history_row(position=1, error=1.0, energy_before=2.0, energy_after=1.0, depth=1, labels=[], skipped=True, nfev=0),
    ]
    ledger = prefix_query_ledger(history, 1)
    assert ledger == {
        "N_H_outer_eval": 2,
        "N_H_refit_eval": 4,
        "N_grad_probe": 6,
        "N_metric_probe": 12,
        "N_other_quantum": 0,
        "S": 24,
    }


def test_structural_prefix_uses_runtime_seed_plus_history_and_tracks_skip() -> None:
    history = [
        _history_row(position=0, error=3.0, energy_before=4.0, energy_after=3.0, depth=1, labels=["A"]),
        _history_row(position=1, error=2.0, energy_before=3.0, energy_after=3.0, depth=1, labels=[], skipped=True),
        _history_row(position=2, error=1.0, energy_before=3.0, energy_after=1.0, depth=2, labels=["B"]),
    ]
    seed = {
        "adapt_vqe": {
            "operators": ["A", "B"],
            "selected_operator_execution_modes": ["termwise_product", "termwise_product"],
            "selected_operator_pauli_terms": [
                [{"pauli_exyz": "xe", "coeff_re": 0.5, "coeff_im": 0.0}],
                [{"pauli_exyz": "ex", "coeff_re": -0.5, "coeff_im": 0.0}],
            ],
            "selected_operator_supports": [[1], [0]],
            "optimal_point": [0.1, 0.2],
        }
    }
    prefix = reconstruct_structural_prefix(seed=seed, history=history, history_position=1)
    assert prefix["selected_labels"] == ["A"]
    assert prefix["selected_generator_count"] == 1
    assert prefix["repeat_skip_count_full_horizon"] == 1
    assert prefix["selected_prefix_parameter_status"] == "blocked_prefix_optimized_parameters_not_serialized"
    assert prefix["selected_generator_semantics"][0]["pauli_terms"][0]["coeff_re"] == 0.5

    terminal = reconstruct_structural_prefix(seed=seed, history=history, history_position=2)
    assert terminal["terminal_parameters_match_selected_structure"] is True
    assert terminal["selected_prefix_theta"] == [0.1, 0.2]


def test_appendix_fragment_places_each_case_table_immediately_after_plot(tmp_path: Path) -> None:
    row = {
        "case_id": "hubbard_L2_scaling_weak",
        "case_title_tex": "Hubbard, L=2, weak",
        "plot": {"pdf": "output/pdf/example/plots/case.pdf"},
        "marker": {"k": 2, "error_raw": 1.0e-5},
        "query_ledger": {"S": 1234},
        "qiskit_prefix_cost": {"status": "ok", "N2q": 10, "D2q": 8, "Dcirc": 20},
        "strict_replay": {"status": "blocked_prefix_optimized_parameters_not_serialized"},
        "cutoff_pair": {"n_ph_work": None, "n_ph_ref": None},
    }
    path = tmp_path / "appendix.tex"
    write_appendix_fragment(path, [row])
    source = path.read_text(encoding="utf-8")
    assert source.count(r"\begin{figure}[H]") == 1
    assert source.count(r"\begin{table}[H]") == 1
    assert source.index(r"\end{figure}") < source.index(r"\begin{table}[H]")
    between = source[source.index(r"\end{figure}") : source.index(r"\begin{table}[H]")]
    assert between.strip() == r"\end{figure}"
    assert "1{,}234" in source
    assert "1\\times 10^{-5}" in source
    assert "blocked_prefix" not in source
    assert "does not serialize its optimized parameters" in source
