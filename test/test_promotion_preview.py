"""Promotion-preview selection and copy-ready table contracts."""

from pathlib import Path

from pipelines.time_dynamics.diagnostics.promotion_preview import (
    _failed_gates,
    _latest_t20_outcomes,
    _latest_t20_records,
    planned_tuple_matrix,
    planned_tuple_matrix_tex,
    preview_pair_tex,
    promotion_preview_pairs,
)


def test_planned_matrix_shows_candidate_values_and_only_missing_arms_pending() -> None:
    item = {
        "drive": "weakslow",
        "controller": "delta_theta_5e-5",
        "methods": {
            "exchange": {
                "paper_status": "pending",
                "latest_evidence_status": "not_run",
                "latest_row": None,
            },
            "avqds": {
                "paper_status": "candidate",
                "latest_evidence_status": "requires_threshold_extension",
                "latest_row": {
                    "mean_infidelity": 4.8e-7,
                    "mean_abs_energy_error": 2.0e-5,
                    "mean_abs_doublon_error": 2.7e-5,
                    "mean_abs_charge_imbalance_error": 3.5e-4,
                    "time_step_count": 341_314,
                    "n_theta": 58,
                    "graph": [168, 230, 362],
                },
            },
        },
    }

    tex = planned_tuple_matrix_tex([item])

    assert "reference &" not in tex
    assert "full APM & \\multicolumn{7}" in tex
    assert "AVQDS &" in tex
    assert "evidence &" not in tex
    assert r"4.8\!\times\!10^{-7}" in tex
    assert "341,314" in tex
from pipelines.time_dynamics.staged_campaign import (
    AccuracyMetrics,
    AccuracyPolicy,
    CampaignLedger,
    CellKey,
    RungRecord,
    STATUS_SAFEGUARD_TRIPPED,
)


def test_response_failed_gates_use_target_normalized_q_not_raw_r() -> None:
    key = CellKey("hh", "weakslow", "avqds", "delta_theta_5e-5")
    policy = AccuracyPolicy.response_normalized_matched_pair(
        response_scales={
            "weakslow": {
                horizon: {
                    "energy_rms_excursion": 0.1,
                    "doublon_rms_excursion": 0.1,
                    "max_site_occupation_rms_excursion": 0.1,
                }
                for horizon in ("t5", "t20")
            }
        }
    )
    record = RungRecord.completed(
        key=key,
        horizon="t20",
        threshold=1.0e-3,
        run_path="output/t20/run.json",
        metrics=AccuracyMetrics(0.004, 0.006, 0.002, 5.0e-5),
        policy=policy,
        resume_from_run_path="output/t5/run.json",
    )
    failures = dict(_failed_gates(record, policy))
    assert r"$q_E$" not in failures
    assert failures[r"$q_D$"] == 1.2


def _record(method: str, threshold: float, run_path: str) -> RungRecord:
    return RungRecord.completed(
        key=CellKey("hh", "weakslow", method, "delta_theta_5e-5"),
        horizon="t20",
        threshold=threshold,
        run_path=run_path,
        metrics=AccuracyMetrics(2.0e-5, 3.0e-5, 1.5e-4, 4.0e-6),
        policy=AccuracyPolicy(),
        resume_from_run_path=run_path.replace("t20", "t5"),
    )


def test_preview_selects_latest_evaluated_record_per_method(monkeypatch) -> None:
    ledger = CampaignLedger()
    for record in (
        _record("avqds", 3.0e-6, "/tmp/avqds-old-t20.json"),
        _record("exchange", 3.0e-6, "/tmp/apm-old-t20.json"),
        _record("avqds", 1.0e-6, "/tmp/avqds-new-t20.json"),
    ):
        ledger = ledger.append(record)

    monkeypatch.setattr(
        "pipelines.time_dynamics.diagnostics.promotion_preview._read_json",
        lambda _path: {"run_lock": {}},
    )
    monkeypatch.setattr(
        "pipelines.time_dynamics.diagnostics.promotion_preview.assert_comparable",
        lambda _locks: None,
    )

    pairs = promotion_preview_pairs(ledger)

    assert len(pairs) == 1
    assert pairs[0]["records"]["avqds"].threshold == 1.0e-6
    assert pairs[0]["records"]["exchange"].threshold == 3.0e-6


def test_latest_t20_record_ignores_later_provenance_only_safeguard() -> None:
    evaluated = _record("avqds", 1.0e-6, "/tmp/avqds-new-t20.json")
    safeguard = RungRecord(
        key=evaluated.key,
        horizon="t20",
        threshold=5.0e-7,
        status=STATUS_SAFEGUARD_TRIPPED,
        reason="next threshold is below the registered minimum",
    )
    ledger = CampaignLedger().append(evaluated).append(safeguard)

    latest = _latest_t20_records(ledger)

    assert latest[("hh", "weakslow", "delta_theta_5e-5", "avqds")] == evaluated


def test_ladder_exhaustion_is_unresolved_and_retains_last_numeric_row(
    monkeypatch,
) -> None:
    evaluated = _record("avqds", 1.0e-6, "/tmp/avqds-new-t20.json")
    safeguard = RungRecord(
        key=evaluated.key,
        horizon="t20",
        threshold=5.0e-7,
        status=STATUS_SAFEGUARD_TRIPPED,
        reason="next threshold is below the registered minimum",
    )
    ledger = CampaignLedger().append(evaluated).append(safeguard)
    row = {
        "method": "avqds",
        "status": "requires_threshold_extension",
        "mean_infidelity": 4.0e-6,
    }
    monkeypatch.setattr(
        "pipelines.time_dynamics.diagnostics.promotion_preview._exact_references_by_drive",
        lambda **_kwargs: {
            ("hh", "weakslow"): {"path": "/tmp/exact.json", "sha256": "a" * 64}
        },
    )
    monkeypatch.setattr(
        "pipelines.time_dynamics.diagnostics.promotion_preview._preview_row",
        lambda _record, _policy: row,
    )

    outcomes = _latest_t20_outcomes(ledger)
    matrix = planned_tuple_matrix(
        ledger=ledger,
        seed_path=Path("/tmp/seed.json"),
        store=object(),
        reference_substeps=40,
    )

    identity = ("hh", "weakslow", "delta_theta_5e-5", "avqds")
    assert outcomes[identity] == safeguard
    slot = matrix[0]["methods"]["avqds"]
    assert slot["paper_status"] == "unresolved"
    assert slot["state_reason"] == "ladder_or_safeguard_exhausted"
    assert slot["latest_row"] == row


def test_preview_table_is_copy_ready_but_keeps_gate_status_visible() -> None:
    rows = {
        "exchange": {
            "mean_infidelity": 3.39e-6,
            "mean_abs_energy_error": 3.92e-5,
            "mean_abs_doublon_error": 2.17e-4,
            "mean_abs_charge_imbalance_error": 2.38e-4,
            "time_step_count": 123456,
            "n_theta": 40,
            "graph": [140, 200, 298],
            "failed_gates": [
                {"label": r"$\overline{\max_i|\Delta n_i|}$", "value": 1.56e-4}
            ],
        },
        "avqds": {
            "mean_infidelity": 4.99e-7,
            "mean_abs_energy_error": 2.00e-5,
            "mean_abs_doublon_error": 2.66e-5,
            "mean_abs_charge_imbalance_error": 3.46e-4,
            "time_step_count": 65432,
            "n_theta": 56,
            "graph": [156, 218, 340],
            "failed_gates": [
                {"label": r"$\overline{\max_i|\Delta n_i|}$", "value": 1.53e-4}
            ],
        },
    }
    tex = preview_pair_tex(
        pair={"drive": "weakslow", "controller": "delta_theta_5e-5"},
        rows=rows,
        figure_path=Path("plots/preview.png"),
    )

    assert "verified exact-evaluated comparison evidence" in tex
    assert r"3.4\!\times\!10^{-6}" in tex
    assert r"3.9\!\times\!10^{-5}" in tex
    assert r"2.2\!\times\!10^{-4}" in tex
    assert r"2.4\!\times\!10^{-4}" in tex
    assert "123,456" in tex
    assert "full APM" in tex
    assert "AVQDS" in tex
    assert "insertion McLachlan (IM)" in tex
    assert "t=20 row pending" in tex
    assert r"\tau_{L^2}" not in tex
    assert "Current qualification gap" in tex
