from __future__ import annotations

from pipelines.reporting.build_paper_i_geo_l2_main_update import (
    REGIME_ORDER,
    _active_append_cells,
    _geo_sources,
    _marker_y,
)


def test_geo_l2_inventory_resolves_exactly_six_ordered_regimes() -> None:
    sources = _geo_sources()
    assert tuple(sources) == REGIME_ORDER
    assert all(path.is_file() for path in sources.values())


def test_preserved_append_cells_follow_active_main_figure_order() -> None:
    cells = _active_append_cells()
    assert tuple(cells) == REGIME_ORDER
    assert cells["weak-weak"]["k_pl"] == 23
    assert cells["strong-strong"]["k_pl"] == 8


def test_marker_y_accepts_display_row_and_csv_point_shapes() -> None:
    assert _marker_y(
        {
            "k_pl": 2,
            "trajectory_points": [
                {"k": 1, "error": 0.5},
                {"k": 2, "error": 0.25},
            ],
        }
    ) == 0.25
    assert _marker_y(
        {
            "k_pl": 2,
            "trajectory_points": [
                {"k": 1, "abs_delta_e": 0.5},
                {"k": 2, "abs_delta_e": 0.25},
            ],
        }
    ) == 0.25
