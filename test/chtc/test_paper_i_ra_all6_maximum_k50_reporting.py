from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.ra_adapt.contracts import canonical_sha256


REPORTING_PATH = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_all6_maximum_k50_reporting_20260817.py"
)
CAMPAIGN_ID = (
    "paper_i_ra_all6_adaptive_shortlist_append_then_plateau_maximum_k50_"
    "20260817_v1"
)
REGIMES = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
NATURAL_KIND = "authenticated_phase3_no_positive_natural_terminal_v1"
NATURAL_OUTCOME = "phase_iii_no_positive_feasible_candidate_v1"


def _load_reporting():
    spec = importlib.util.spec_from_file_location(
        "paper_i_ra_all6_maximum_k50_reporting",
        REPORTING_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _signed(payload: dict[str, Any]) -> dict[str, Any]:
    return {**copy.deepcopy(payload), "sha256": canonical_sha256(payload)}


def _accepted_row(
    *,
    identity: dict[str, Any],
    controller_round: int,
    placement_state: str,
    energy: float,
) -> dict[str, Any]:
    return {
        **identity,
        "controller_round": controller_round,
        "energy": energy,
        "absolute_delta_e": abs(energy + 2.0),
        "placement_state": placement_state,
        "phase0_population_count": 8,
        "phase0_retained_count": 6,
        "phase_i_input_count": 6,
        "phase_i_retained_count": 4,
        "phase_ii_input_count": 4,
        "phase_ii_retained_count": 3,
        "phase_iii_input_count": 3,
        "phase_iii_adaptive_retained_count": 2,
        "phase_iii_final_singleton_count": 1,
        "phase_iii_final_record_id": (
            f"record::{identity['regime_id']}::{controller_round}"
        ),
        "selected_generator": f"generator::{controller_round}",
        "selected_operator": f"operator::{controller_round}",
        "selected_position": controller_round - 1,
        "s_alg": 100 * controller_round,
        "n2q": 10 * controller_round,
        "d2q": 7 * controller_round,
        "dc": 12 * controller_round,
        "checkpoint_sha256": f"{identity['cell_ordinal']:02x}" * 32,
    }


def _terminal_attempt(
    *, identity: dict[str, Any], attempted_round: int, placement_state: str
) -> dict[str, Any]:
    return {
        **identity,
        "attempted_controller_round": attempted_round,
        "terminal_controller_outcome": NATURAL_OUTCOME,
        "placement_state": placement_state,
        "phase0_population_count": 8,
        "phase0_retained_count": 6,
        "phase_i_input_count": 6,
        "phase_i_retained_count": 4,
        "phase_ii_input_count": 4,
        "phase_ii_retained_count": 3,
        "phase_iii_input_count": 3,
        "phase_iii_adaptive_retained_count": 0,
        "phase_iii_final_singleton_count": 0,
        "terminal_phase3_selection_receipt_sha256": (
            f"{identity['cell_ordinal'] + 32:02x}" * 32
        ),
        "terminal_active_prefix_checkpoint_sha256": (
            f"{identity['cell_ordinal'] + 64:02x}" * 32
        ),
    }


def _cell_projection(
    *,
    block: str,
    regime_index: int,
    accepted_rounds: int,
    open_accepted_rounds: tuple[int, ...] = (),
    terminal_open: bool = False,
) -> dict[str, Any]:
    regime_id, nph = REGIMES[regime_index]
    insertion_policy = (
        "append_only" if block == "append" else "plateau_commutation"
    )
    cell_ordinal = regime_index + 1 + (0 if block == "append" else 6)
    identity = {
        "execution_id": (
            "all_phase_adaptive_natural_terminal__"
            f"{regime_id}__nph{nph}__{insertion_policy}__maximum_k50"
        ),
        "cell_ordinal": cell_ordinal,
        "block": block,
        "regime_id": regime_id,
        "nph": nph,
        "insertion_policy": insertion_policy,
    }
    accepted_rows = [
        _accepted_row(
            identity=identity,
            controller_round=controller_round,
            placement_state=(
                "append_only"
                if block == "append"
                else (
                    "open"
                    if controller_round in open_accepted_rounds
                    else "closed"
                )
            ),
            energy=-1.0 - 0.1 * controller_round + (
                0.05 if block == "plateau" else 0.0
            ),
        )
        for controller_round in range(1, accepted_rounds + 1)
    ]
    attempted_round = accepted_rounds + 1
    completion = _signed(
        {
            "schema": (
                "paper_i_ra_all6_adaptive_maximum_k50_cell_completion_v1"
            ),
            "campaign_id": CAMPAIGN_ID,
            "execution_id": identity["execution_id"],
            "cell_ordinal": cell_ordinal,
            "completion_kind": NATURAL_KIND,
            "maximum_controller_rounds": 50,
            "accepted_controller_rounds": accepted_rounds,
            "terminal_attempted_controller_round": attempted_round,
            "terminal_controller_outcome": NATURAL_OUTCOME,
            "terminal_phase3_selection_receipt_sha256": (
                f"{cell_ordinal + 32:02x}" * 32
            ),
            "summary_artifact_status": (
                "not_applicable_round_zero" if accepted_rounds == 0 else "present"
            ),
        }
    )
    projection = {
        "schema": "paper_i_ra_all6_maximum_k50_reporting_cell_projection_v1",
        "cell": identity,
        "completion": completion,
        "accepted_rows": accepted_rows,
        "terminal_attempt": _terminal_attempt(
            identity=identity,
            attempted_round=attempted_round,
            placement_state=(
                "append_only"
                if block == "append"
                else ("open" if terminal_open else "closed")
            ),
        ),
        "archive_backed_closure_sha256": f"{cell_ordinal + 96:02x}" * 32,
    }
    return _signed(projection)


def _matrix() -> list[dict[str, Any]]:
    append_rounds = (2, 1, 0, 2, 1, 1)
    plateau_rounds = (1, 1, 0, 2, 0, 2)
    rows: list[dict[str, Any]] = []
    for block, counts in (
        ("append", append_rounds),
        ("plateau", plateau_rounds),
    ):
        for regime_index, accepted_rounds in enumerate(counts):
            rows.append(
                _cell_projection(
                    block=block,
                    regime_index=regime_index,
                    accepted_rounds=accepted_rounds,
                    open_accepted_rounds=(1,)
                    if block == "plateau" and regime_index == 0
                    else (),
                    terminal_open=(block == "plateau" and regime_index == 4),
                )
            )
    return rows


def test_build_ragged_report_keeps_terminal_attempts_outside_accepted_rows() -> None:
    reporting = _load_reporting()

    report = reporting.build_ragged_report(list(reversed(_matrix())))

    assert report["schema"] == (
        "paper_i_ra_all6_adaptive_maximum_k50_ragged_comparison_v1"
    )
    assert report["status"] == "passed_all6_append_then_plateau_maximum_k50"
    assert report["accepted_row_count"] == 13
    assert [row["controller_round"] for row in report["accepted_rows"][:3]] == [
        1,
        2,
        1,
    ]
    assert len(report["terminal_attempts"]) == 12
    assert report["terminal_attempts"][2]["attempted_controller_round"] == 1
    assert "selected_generator" not in report["terminal_attempts"][2]
    assert "n2q" not in report["terminal_attempts"][2]
    assert report["sha256"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "sha256"}
    )


def test_pairing_uses_common_prefix_and_never_compares_ragged_endpoints() -> None:
    reporting = _load_reporting()

    report = reporting.build_ragged_report(_matrix())

    assert len(report["shared_prefix_pairs"]) == 5
    weak_pair = report["shared_prefix_pairs"][0]
    assert weak_pair["regime_id"] == "weak_weak"
    assert weak_pair["controller_round"] == 1
    assert weak_pair["selected_record_status"] == "agree"
    assert weak_pair["plateau_minus_append_energy"] == pytest.approx(0.05)
    assert weak_pair["plateau_minus_append_s_alg"] == 0

    endpoints = {
        row["regime_id"]: row for row in report["regime_endpoint_pairs"]
    }
    weak_endpoint = endpoints["weak_weak"]
    assert weak_endpoint["comparison_status"] == "not_compared"
    assert weak_endpoint["null_reason"] == (
        "accepted_controller_round_mismatch"
    )
    assert weak_endpoint["controller_round"] is None
    assert weak_endpoint["plateau_minus_append_energy"] is None

    equal_endpoint = endpoints["intermediate_weak"]
    assert equal_endpoint["comparison_status"] == "compared"
    assert equal_endpoint["null_reason"] is None
    assert equal_endpoint["controller_round"] == 1
    assert equal_endpoint["plateau_minus_append_energy"] == pytest.approx(0.05)

    zero_endpoint = endpoints["strong_weak_u8"]
    assert zero_endpoint["comparison_status"] == "not_compared"
    assert zero_endpoint["null_reason"] == "no_accepted_controller_rounds"
    assert zero_endpoint["controller_round"] is None

    outcomes = {
        (row["block"], row["regime_id"]): row
        for row in report["cell_outcomes"]
    }
    assert outcomes[("append", "weak_weak")][
        "placement_activation_status"
    ] == "append_only"
    assert outcomes[("plateau", "weak_weak")][
        "placement_activation_status"
    ] == "activated_with_accepted_transition"
    assert outcomes[("plateau", "intermediate_strong")][
        "placement_activation_status"
    ] == "activated_terminal_attempt_only"
    assert outcomes[("plateau", "intermediate_weak")][
        "placement_activation_status"
    ] == "not_activated"


def test_renderers_are_deterministic_lf_only_and_safe_with_zero_accepted_rows() -> None:
    reporting = _load_reporting()
    zero_matrix = [
        _cell_projection(
            block=block,
            regime_index=regime_index,
            accepted_rounds=0,
        )
        for block in ("append", "plateau")
        for regime_index in range(6)
    ]
    report = reporting.build_ragged_report(zero_matrix)

    json_text = reporting.render_report_json(report)
    accepted_csv = reporting.render_accepted_rows_csv(report)
    terminal_csv = reporting.render_terminal_attempts_csv(report)
    pairs_csv = reporting.render_shared_prefix_pairs_csv(report)
    markdown = reporting.render_markdown(report)

    assert json.loads(json_text) == report
    assert json_text == reporting.render_report_json(report)
    assert json_text.endswith("\n") and "\r" not in json_text
    assert accepted_csv == (
        "execution_id,cell_ordinal,block,regime_id,nph,insertion_policy,"
        "controller_round,energy,absolute_delta_e,placement_state,"
        "phase0_population_count,phase0_retained_count,phase_i_input_count,"
        "phase_i_retained_count,phase_ii_input_count,phase_ii_retained_count,"
        "phase_iii_input_count,phase_iii_adaptive_retained_count,"
        "phase_iii_final_singleton_count,phase_iii_final_record_id,"
        "selected_generator,selected_operator,selected_position,s_alg,n2q,d2q,"
        "dc,checkpoint_sha256\n"
    )
    assert pairs_csv.count("\n") == 1
    assert terminal_csv.count("\n") == 13
    assert "selected_generator" not in terminal_csv.splitlines()[0]
    assert all("\r" not in value for value in (accepted_csv, terminal_csv, pairs_csv))
    assert "No accepted controller rounds were recorded." in markdown
    assert "Accepted rows: **0**" in markdown
    assert markdown.endswith("\n") and "\r" not in markdown


def test_terminal_attempt_must_match_the_authenticated_completion_receipt() -> None:
    reporting = _load_reporting()
    cells = _matrix()
    cells[0]["terminal_attempt"][
        "terminal_phase3_selection_receipt_sha256"
    ] = "ff" * 32
    cells[0]["sha256"] = canonical_sha256(
        {key: value for key, value in cells[0].items() if key != "sha256"}
    )

    with pytest.raises(
        reporting.ReportingError,
        match="Terminal Phase-III receipt binding drifted",
    ):
        reporting.build_ragged_report(cells)


def test_projection_digest_detects_accepted_row_tamper() -> None:
    reporting = _load_reporting()
    cells = _matrix()
    cells[0]["accepted_rows"][0]["energy"] = -999.0

    with pytest.raises(
        reporting.ReportingError, match="reporting cell projection digest drifted"
    ):
        reporting.build_ragged_report(cells)


def test_resigned_projection_cannot_hide_accepted_cardinality_drift() -> None:
    reporting = _load_reporting()
    cells = _matrix()
    cells[0]["accepted_rows"].pop()
    cells[0]["sha256"] = canonical_sha256(
        {key: value for key, value in cells[0].items() if key != "sha256"}
    )

    with pytest.raises(reporting.ReportingError, match="completion binding drifted"):
        reporting.build_ragged_report(cells)


def test_terminal_domain_rejects_selected_or_compiled_resource_fields() -> None:
    reporting = _load_reporting()
    cells = _matrix()
    cells[0]["terminal_attempt"]["selected_generator"] = "smuggled"
    cells[0]["terminal_attempt"]["n2q"] = 1
    cells[0]["sha256"] = canonical_sha256(
        {key: value for key, value in cells[0].items() if key != "sha256"}
    )

    with pytest.raises(
        reporting.ReportingError, match="terminal attempt fields drifted"
    ):
        reporting.build_ragged_report(cells)


def test_exact_matrix_rejects_duplicate_cell_even_when_projection_is_valid() -> None:
    reporting = _load_reporting()
    cells = _matrix()
    cells[-1] = copy.deepcopy(cells[0])

    with pytest.raises(reporting.ReportingError, match="exact all-six matrix"):
        reporting.build_ragged_report(cells)


def test_terminal_only_activation_is_preserved_without_an_accepted_transition() -> None:
    reporting = _load_reporting()
    cells = [
        _cell_projection(
            block=block,
            regime_index=regime_index,
            accepted_rounds=0,
            terminal_open=(block == "plateau" and regime_index == 0),
        )
        for block in ("append", "plateau")
        for regime_index in range(6)
    ]

    report = reporting.build_ragged_report(cells)

    assert report["placement_factor_status"] == "activated_terminal_attempt_only"
    plateau_weak = next(
        row
        for row in report["cell_outcomes"]
        if row["block"] == "plateau" and row["regime_id"] == "weak_weak"
    )
    assert plateau_weak["placement_activation_status"] == (
        "activated_terminal_attempt_only"
    )


def _maximum_cell_projection(
    *, block: str, regime_index: int
) -> dict[str, Any]:
    projection = _cell_projection(
        block=block,
        regime_index=regime_index,
        accepted_rounds=50,
    )
    completion = projection["completion"]
    completion["completion_kind"] = "reached_maximum_controller_rounds_v1"
    completion["terminal_attempted_controller_round"] = None
    completion["terminal_controller_outcome"] = None
    completion["terminal_phase3_selection_receipt_sha256"] = None
    completion["sha256"] = canonical_sha256(
        {key: value for key, value in completion.items() if key != "sha256"}
    )
    projection["terminal_attempt"] = None
    projection["sha256"] = canonical_sha256(
        {key: value for key, value in projection.items() if key != "sha256"}
    )
    return projection


def test_exact_k50_cells_have_no_fabricated_terminal_attempts() -> None:
    reporting = _load_reporting()
    cells = [
        _maximum_cell_projection(block=block, regime_index=regime_index)
        for block in ("append", "plateau")
        for regime_index in range(6)
    ]

    report = reporting.build_ragged_report(cells)

    assert report["accepted_row_count"] == 600
    assert report["terminal_attempt_count"] == 0
    assert report["terminal_attempts"] == []
    assert len(report["shared_prefix_pairs"]) == 300
    assert all(
        row["comparison_status"] == "compared"
        and row["controller_round"] == 50
        for row in report["regime_endpoint_pairs"]
    )
