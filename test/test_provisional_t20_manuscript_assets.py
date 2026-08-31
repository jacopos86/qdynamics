"""Paper-II full-horizon manuscript-asset contracts."""

from pipelines.time_dynamics.diagnostics.provisional_t20_manuscript_assets import (
    _table_tex,
)


def test_table_reports_one_compact_infidelity_resource_row_per_run() -> None:
    table = _table_tex(
        [
            {
                "drive": "weakslow",
                "controller": "delta_theta_5e-5",
                "method": "apm",
                "paper_status": "qualified",
                "cut": "3e-06",
                "mean_abs_energy_error": 3.92e-5,
                "mean_abs_doublon_error": 2.17e-4,
                "mean_abs_charge_imbalance_error": 2.38e-4,
                "mean_infidelity": 4.8e-6,
                "time_step_count": 487_321,
                "n_theta": 40,
                "graph": [140, 200, 298],
            }
        ]
    )

    assert r"\overline{1-F}" in table
    assert r"4.8\!\times\!10^{-6}" in table
    assert r"3.9\!\times\!10^{-5}" in table
    assert r"2.2\!\times\!10^{-4}" in table
    assert r"2.4\!\times\!10^{-4}" in table
    assert "487,321" in table
    assert "insertion McLachlan (IM)" in table
    assert "$t=20$ matrix pending" in table
    assert r"\tau_{L^2}" not in table
    assert "evidence" not in table
    assert "compiled" not in table
    assert "RHS" not in table
    assert "time-step controller" in table
    assert "campaign results PDF" in table
    assert "Additional observable plots remain" in table
    assert "aggregate accuracy burdens" in table
    assert "every deadbanded primitive $q$ component" in table


def test_table_shows_completed_candidates_and_keeps_only_unfinished_slots_pending() -> None:
    table = _table_tex(
        [
            {
                "drive": "fastweak",
                "controller": "state_motion_5e-5",
                "method": "apm",
                "paper_status": "pending",
                "pending_reason": "t20_not_completed",
            },
            {
                "drive": "fastweak",
                "controller": "state_motion_5e-5",
                "method": "avqds",
                "paper_status": "candidate",
                "pending_reason": "accuracy_gate_pending_threshold_extension",
                "mean_infidelity": 4.8e-7,
                "mean_abs_energy_error": 2.0e-5,
                "mean_abs_doublon_error": 2.7e-5,
                "mean_abs_charge_imbalance_error": 3.5e-4,
                "time_step_count": 341_314,
                "n_theta": 58,
                "graph": [168, 230, 362],
            },
        ]
    )

    assert "cached" not in table
    assert "state motion & full APM & \\multicolumn{7}" in table
    assert "state motion & AVQDS &" in table
    assert "candidate" not in table
    assert "run not completed" in table
    assert r"4.8\!\times\!10^{-7}" in table


def test_table_keeps_unresolved_numeric_evidence_and_marks_empty_exhaustion() -> None:
    table = _table_tex(
        [
            {
                "drive": "midres",
                "controller": "state_motion_5e-5",
                "method": "apm",
                "paper_status": "unresolved",
                "state_reason": "ladder_or_safeguard_exhausted",
            },
            {
                "drive": "midres",
                "controller": "state_motion_5e-5",
                "method": "avqds",
                "paper_status": "unresolved",
                "state_reason": "ladder_or_safeguard_exhausted",
                "mean_infidelity": 7.0e-5,
                "mean_abs_energy_error": 2.0e-4,
                "mean_abs_doublon_error": 3.0e-4,
                "mean_abs_charge_imbalance_error": 4.0e-4,
                "time_step_count": 99,
                "n_theta": 12,
                "graph": [1, 2, 3],
            },
        ]
    )

    assert "unresolved: ladder/resource exhausted" in table
    assert r"7.0\!\times\!10^{-5}" in table
