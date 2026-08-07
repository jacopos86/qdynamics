from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.reporting.build_paper_i_hh_corrected_parent_comparator_page13_pdf import (
    _snake_curve_from_source,
    active_page13_snake_cells,
    first_plateau_history_row,
    generic_curve,
    latex_graphics_path,
    prefix_query_ledger,
)


def _history_row(
    *,
    position: int,
    error: float,
    depth: int,
    energy_before: float,
    outer: int,
    refit: int,
    grad: int,
    metric: int,
) -> dict[str, object]:
    return {
        "history_position": position,
        "energy_before": energy_before,
        "energy_after": error,
        "abs_delta_e_same_cutoff_after": error,
        "depth_after": depth,
        "outer_hamiltonian_eval_count": outer,
        "optimizer_nfev": refit,
        "selector_gradient_probe_count": grad,
        "selector_metric_probe_count": metric,
        "qngd_gradient_operator_probe_count_total": 0,
        "qngd_metric_operator_probe_count_total": 0,
    }


def test_generic_plateau_keeps_outer_iteration_distinct_from_geo_depth_and_charges_skip_work() -> None:
    payload = {
        "result": {
            "same_cutoff_exact_gs_energy": 0.0,
            "adapt_history": [
                _history_row(
                    position=0,
                    error=1.0,
                    depth=1,
                    energy_before=2.0,
                    outer=1,
                    refit=3,
                    grad=5,
                    metric=7,
                ),
                _history_row(
                    position=1,
                    error=0.50,
                    depth=1,
                    energy_before=1.0,
                    outer=1,
                    refit=11,
                    grad=13,
                    metric=17,
                ),
                _history_row(
                    position=2,
                    error=0.49,
                    depth=2,
                    energy_before=0.50,
                    outer=1,
                    refit=19,
                    grad=23,
                    metric=29,
                ),
            ],
        }
    }
    skipped = payload["result"]["adapt_history"][1]
    skipped.update(
        geo_immediate_repeat_skipped=True,
        appended_operator_count=0,
        selected_batch_labels=[],
    )

    curve = generic_curve(payload)
    assert [(point.k, point.error) for point in curve] == [
        (0, 2.0),
        (1, 1.0),
        (2, 0.50),
        (3, 0.49),
    ]
    assert first_plateau_history_row(payload) == (1, 2, 1, 0.50)

    ledger = prefix_query_ledger(payload["result"]["adapt_history"], 1)
    assert ledger == {
        "N_H_outer": 2,
        "N_H_refit": 14,
        "N_grad_selector": 18,
        "N_grad_qngd": 0,
        "N_metric_selector": 24,
        "N_metric_qngd": 0,
        "N_other_quantum": 0,
        "S_alg": 58,
    }


def test_plateau_horizon_is_a_display_selection_crop_not_a_history_truncation() -> None:
    history = [
        _history_row(
            position=index,
            error=error,
            depth=index + 1,
            energy_before=2.0 if index == 0 else 1.0,
            outer=1,
            refit=1,
            grad=1,
            metric=1,
        )
        for index, error in enumerate((1.0, 0.5, 0.49, 0.10, 0.095))
    ]
    payload = {
        "result": {
            "same_cutoff_exact_gs_energy": 0.0,
            "adapt_history": history,
        }
    }

    assert first_plateau_history_row(payload, max_iterations=3) == (1, 2, 2, 0.5)
    assert first_plateau_history_row(payload) == (3, 4, 4, 0.10)
    assert len(payload["result"]["adapt_history"]) == 5

    with pytest.raises(ValueError, match="exceeds history length"):
        first_plateau_history_row(payload, max_iterations=6)


def test_active_page13_snake_cells_include_corrected_visible_s() -> None:
    cells = active_page13_snake_cells()
    assert list(cells) == [
        "weak-weak",
        "intermediate-weak",
        "strong-weak",
        "weak-strong",
        "intermediate-strong",
        "strong-strong",
    ]
    assert cells["weak-weak"]["S_alg"] == 5009
    assert cells["strong-strong"]["S_alg"] == 5153


def test_latex_graphics_path_detokenizes_safe_paths_and_rejects_percent() -> None:
    assert latex_graphics_path(Path("safe path/plot_1.pdf")) == r"\detokenize{safe path/plot_1.pdf}"
    with pytest.raises(ValueError, match="TeX-unsafe"):
        latex_graphics_path(Path("bad%path/plot.pdf"))


@pytest.mark.parametrize("stitched", [False, True])
def test_snake_curve_uses_reference_state_error_at_k_zero(tmp_path: Path, stitched: bool) -> None:
    table_source = tmp_path / "snake_table.json"
    table_source.write_text(
        json.dumps(
            {
                "adapt_vqe": {
                    "history": [
                        {"delta_abs_prev": 2.5, "delta_abs_current": 1.2},
                        {"delta_abs_prev": 1.2, "delta_abs_current": 0.4},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    curve_source = tmp_path / "snake_curve.json"
    if stitched:
        payload = {"points": [{"k": 0, "abs_delta_e": 1.2}, {"k": 1, "abs_delta_e": 0.4}]}
    else:
        payload = json.loads(table_source.read_text(encoding="utf-8"))
    curve_source.write_text(json.dumps(payload), encoding="utf-8")

    points = _snake_curve_from_source(curve_source, table_source)
    assert points[0].k == 0
    assert points[0].error == 2.5
    if stitched:
        assert [(point.k, point.error) for point in points] == [(0, 2.5), (1, 0.4)]
    else:
        assert [(point.k, point.error) for point in points] == [(0, 2.5), (1, 1.2), (2, 0.4)]
