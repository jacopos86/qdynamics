"""The compiled-Q exchange route on SIAM ``logical_shared`` seeds.

SIAM starts from an empty ansatz, so the state's executor carries no qubit
count.  Every compiled-cost bundle used to raise at the first checkpoint
(``int(None)``) before a single word was compiled.  These tests pin the qubit
count down to the reference-state dimension, the route label that distinguishes
a compiled-Q run from the historical route, and the runner fields the SIAM
scoring and stitching stack reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.ap_mclachlan import compiled_patch_cost as cpc
from pipelines.time_dynamics.ap_mclachlan.state import (
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.diagnostics.siam_segment_stitching import _row_costs
from pipelines.time_dynamics.paper_ii_runs import build_siam_nb2_proof_run
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import main
from pipelines.time_dynamics.siam_screening import SIAMScreeningConfig
from pipelines.time_dynamics.siam_seed import (
    load_siam_runtime_input,
    write_siam_seed_manifest,
)

# Keys ``siam_offline_scoring`` reads off a raw run's ``summary``.
SCORING_SUMMARY_KEYS = ("uncured_local_subdivision_count",)

# Plot-row keys ``siam_segment_stitching._row_costs`` reads via ``row.get``.
STITCHING_ROW_KEYS = (
    "patch_commit_count",
    "patch_commit_kinds",
    "patch_append_commit_count",
    "patch_append_inserted_coordinate_count",
    "patch_delete_commit_count",
    "patch_delete_deleted_coordinate_count",
    "patch_exchange_commit_count",
    "patch_exchange_inserted_coordinate_count",
    "patch_exchange_deleted_coordinate_count",
)


def _seed(path: Path, *, n_bath: int = 2) -> None:
    write_siam_seed_manifest(
        path,
        SIAMScreeningConfig(
            n_bath=n_bath,
            u=1.5,
            t_final=99.0,
            num_times=2,
            label="compiled_q_route_regression",
        ),
    )


def _siam_state(path: Path):
    return state_from_scaffold_runtime_input(
        load_siam_runtime_input(path),
        parameterization_mode="logical_shared",
    )


def test_siam_seed_state_has_no_executor_qubit_count(tmp_path: Path) -> None:
    """The precondition the compiled-cost runtime used to trip over."""

    seed = tmp_path / "siam_seed.json"
    _seed(seed)
    state = _siam_state(seed)

    assert state.runtime_parameter_count == 0
    assert getattr(state.executor, "nq", None) is None


def test_state_num_qubits_falls_back_to_the_reference_dimension(
    tmp_path: Path,
) -> None:
    """``CompiledCostRuntime.__init__`` reads the qubit count through this."""

    import numpy as np

    seed = tmp_path / "siam_seed.json"
    _seed(seed)
    state = _siam_state(seed)

    dimension = int(np.asarray(state.psi_ref, dtype=complex).reshape(-1).size)
    nq = cpc.state_num_qubits(state)

    assert (1 << nq) == dimension
    # Nb=2 SIAM is three sites, two spins: six qubits.
    assert nq == 6


def _run(tmp_path: Path, arm: str, name: str) -> dict:
    seed = tmp_path / "siam_seed.json"
    output = tmp_path / f"{name}.json"
    _seed(seed)
    run = build_siam_nb2_proof_run(
        seed_path=str(seed),
        arm=arm,
        output_json=str(output),
        code_revision="compiled-q-route-test",
        extra_flags=("--t-final", "0.08", "--num-times", "3", "--compile-workers", "1"),
    )
    assert main(list(run.argv())) == 0
    return json.loads(output.read_text(encoding="utf-8"))


def test_compiled_q_arm_completes_a_siam_logical_shared_run(tmp_path: Path) -> None:
    """Regression: the compiled-cost bundle used to raise before any checkpoint."""

    payload = _run(tmp_path, "exchange_q_compiled", "qc")

    assert payload["run_lock"]["physics"]["family_key"] == "siam"
    points = payload["trajectory"]["points"]
    assert len(points) == 3
    # Compiled scoring actually ran: the route committed structure.
    assert any(bool(p["patch_decision"]["accepted"]) for p in points)


def test_compiled_q_run_reports_the_route_bundle_in_force(tmp_path: Path) -> None:
    """``selection_policy`` names the family; the bundle names the route."""

    payload = _run(tmp_path, "exchange_q_compiled", "qc")
    trajectory = payload["trajectory"]

    metadata = trajectory["metadata"]
    assert metadata["selection_policy"] == "paper_ii_generalized_exchange_v2"
    assert metadata["selection_route_bundle"] == "paper_ii_exchange_q_compiled_v1"

    decided = [
        point["patch_decision"]["metadata"]
        for point in trajectory["points"]
        if point["patch_decision"].get("metadata")
        and "selection_route_bundle" in point["patch_decision"]["metadata"]
    ]
    assert decided, "no checkpoint reported a selection route bundle"
    for entry in decided:
        assert entry["selection_policy"] == "paper_ii_generalized_exchange_v2"
        assert entry["selection_route_bundle"] == "paper_ii_exchange_q_compiled_v1"
        assert (
            entry["selector_route"]["bundle"]["bundle_id"]
            == entry["selection_route_bundle"]
        )


def test_historical_exchange_run_still_reports_the_historical_bundle(
    tmp_path: Path,
) -> None:
    payload = _run(tmp_path, "exchange", "hist")

    metadata = payload["trajectory"]["metadata"]
    assert metadata["selection_policy"] == "paper_ii_generalized_exchange_v2"
    assert metadata["selection_route_bundle"] == "paper_ii_generalized_exchange_v2"


@pytest.mark.parametrize("arm", ("exchange", "exchange_q_compiled", "avqds"))
def test_runner_emits_every_field_the_siam_stack_reads(
    tmp_path: Path, arm: str
) -> None:
    """Scoring fails closed on a missing key; stitching silently zeroes one."""

    payload = _run(tmp_path, arm, arm)
    summary = payload["summary"]
    rows = payload["plot_rows"]

    for key in SCORING_SUMMARY_KEYS:
        assert key in summary, f"summary is missing {key}"
        assert isinstance(summary[key], int)

    assert rows
    for row in rows:
        for key in STITCHING_ROW_KEYS:
            assert key in row, f"plot row is missing {key}"

    # The stitching reducer runs over these rows and agrees with the runner's
    # own, independently derived summary counters.  A missing plot-row key
    # would show up here as a silent zero rather than an error.
    costs = _row_costs(rows)
    assert costs["uncured_local_subdivision_count"] == int(
        summary["uncured_local_subdivision_count"]
    )
    for key in (
        "accepted_patch_count",
        "accepted_append_count",
        "accepted_delete_count",
        "accepted_exchange_count",
    ):
        assert int(costs[key]) == int(summary[key]), key

    # The coordinate counters are deliberately NOT compared against the
    # summary's ``accepted_*_coordinate_count`` keys.  Those read
    # ``patch_inserted_count``/``patch_removed_count``, which are derived from
    # ``decision.selected_score``; on the exchange route that is ``None``, so
    # the summary reports zero coordinates even where the route committed some.
    # The plot counters read the committed selection instead, which is what the
    # stitched cost table needs.  That pre-existing summary under-count is left
    # alone here; changing it would move the historical route's reported
    # numbers.
    committed_insertions = 0
    committed_removals = 0
    for point in payload["trajectory"]["points"]:
        committed = (point["patch_decision"].get("metadata") or {}).get(
            "committed"
        ) or {}
        committed_insertions += len(committed.get("inserted_selection") or ())
        committed_removals += len(committed.get("removed_runtime_indices") or ())
    if committed_insertions or committed_removals:
        assert (
            int(costs["accepted_appended_coordinate_count"])
            + int(costs["accepted_exchange_appended_coordinate_count"])
            == committed_insertions
        )
        assert (
            int(costs["accepted_deleted_coordinate_count"])
            + int(costs["accepted_exchange_deleted_coordinate_count"])
            == committed_removals
        )


def test_commit_counters_agree_with_the_accepted_patch_kind(tmp_path: Path) -> None:
    """One commit per accepted checkpoint, attributed to that patch kind."""

    payload = _run(tmp_path, "exchange_q_compiled", "qc")

    for row in payload["plot_rows"]:
        accepted = bool(row["patch_accepted"])
        kind = str(row["patch_kind"])
        assert int(row["patch_commit_count"]) == (1 if accepted else 0)
        assert list(row["patch_commit_kinds"]) == ([kind] if accepted else [])
        by_kind = {
            "append": "patch_append_commit_count",
            "insert": "patch_append_commit_count",
            "delete": "patch_delete_commit_count",
            "exchange": "patch_exchange_commit_count",
        }
        for counter in set(by_kind.values()):
            expected = int(accepted and by_kind.get(kind) == counter)
            assert int(row[counter]) == expected, (kind, counter)
