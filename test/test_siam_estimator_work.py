"""Contracts for SIAM logical estimator-work accounting.

The accounting unit is one whole trajectory.  A bare continuation segment is
refused because the offline stitcher retains the later boundary row, so summing
two segments would charge the shared boundary checkpoint twice.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from pipelines.time_dynamics.diagnostics.paper_ii_four_route_comparison import (
    complete_estimator_work,
)
from pipelines.time_dynamics.diagnostics.siam_estimator_work import (
    PHYSICAL_SHOTS_STATUS,
    SIAM_ESTIMATOR_WORK_V1,
    build_siam_estimator_work_record,
    verify_siam_estimator_work_record,
    write_siam_estimator_work_record,
)
from pipelines.time_dynamics.diagnostics.siam_segment_stitching import (
    SIAM_NB4_STITCH_V1,
)
from pipelines.time_dynamics.paper_ii_runs import build_siam_nb2_proof_run
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import main
from pipelines.time_dynamics.siam_screening import SIAMScreeningConfig
from pipelines.time_dynamics.siam_seed import write_siam_seed_manifest


ARMS = ("exchange", "avqds")

# Five points on the canonical 0.05 spacing, the shortest grid that still
# exercises an accepted append, a refused checkpoint, and a terminal row.
SMOKE_FLAGS = ("--t-final", "0.2", "--num-times", "5")


# ---------------------------------------------------------------------------
# Synthetic unit coverage for the relaxed initial-support guard.
# ---------------------------------------------------------------------------


def _synthetic_run(
    *,
    initial_labels: list[str] | None,
    declared_initial: int | None = None,
) -> dict[str, Any]:
    """One minimal APM-shaped run with a single trajectory point."""

    initial_state: dict[str, Any] = {"parameterization_mode": "logical_shared"}
    if initial_labels is not None:
        initial_state["runtime_coordinate_labels"] = list(initial_labels)
    summary: dict[str, Any] = {"rhs_evaluation_count": 2}
    if declared_initial is not None:
        summary["runtime_parameter_count_initial"] = int(declared_initial)
    return {
        "initial_state": initial_state,
        "summary": summary,
        "plot_rows": [
            {
                "runtime_parameter_count": 1,
                "parameter_control_rhs_evaluation_count": 0,
                "integration_rhs_evaluation_count": 2,
            }
        ],
        "trajectory": {
            "points": [
                {
                    "fixed_step": {"support_labels": ["a"]},
                    "patch_decision": {
                        "accepted": False,
                        "reason": "escalation_predicate_false",
                        "metadata": {
                            "insertion_rounds": 0,
                            "attempt_count": 0,
                            "attempts": [],
                        },
                    },
                }
            ]
        },
    }


def test_complete_estimator_work_accepts_a_legitimately_empty_initial_support() -> None:
    result = complete_estimator_work(
        [_synthetic_run(initial_labels=[], declared_initial=0)],
        route="q_compiled",
    )

    # g(0) = 0*(0+1)/2 + 0 + 2 = 2 for the one checkpoint entered with no
    # runtime coordinate at all.
    assert result["logical_checkpoint_base_geometry_work"] == 2
    assert result["checkpoint_base_geometry_evaluation_count"] == 1
    assert result["checkpoint_base_status"] == "exact"
    assert result["logical_total_estimator_work_status"] == "exact"


def test_complete_estimator_work_still_fails_closed_without_the_label_key() -> None:
    with pytest.raises(ValueError, match="initial runtime-coordinate labels"):
        complete_estimator_work(
            [_synthetic_run(initial_labels=None)], route="q_compiled"
        )


def test_complete_estimator_work_rejects_a_non_sequence_label_field() -> None:
    run = _synthetic_run(initial_labels=[])
    run["initial_state"]["runtime_coordinate_labels"] = "abc"

    with pytest.raises(ValueError, match="initial runtime-coordinate labels"):
        complete_estimator_work([run], route="q_compiled")


def test_complete_estimator_work_rejects_unlabelled_declared_coordinates() -> None:
    with pytest.raises(ValueError, match="omits their labels"):
        complete_estimator_work(
            [_synthetic_run(initial_labels=[], declared_initial=3)],
            route="q_compiled",
        )


def test_nonempty_initial_support_still_charges_its_own_geometry() -> None:
    result = complete_estimator_work(
        [_synthetic_run(initial_labels=["a"], declared_initial=1)],
        route="q_compiled",
    )

    # g(1) = 1 + 1 + 2 = 4.
    assert result["logical_checkpoint_base_geometry_work"] == 4


# ---------------------------------------------------------------------------
# Real short SIAM runs.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def siam_smoke_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One Nb=2 five-point run per arm, produced in this test session."""

    root = tmp_path_factory.mktemp("siam_estimator_work")
    seed = root / "siam_seed.json"
    write_siam_seed_manifest(
        seed,
        SIAMScreeningConfig(n_bath=2, label="siam_estimator_work_smoke"),
    )
    produced: dict[str, Path] = {"seed": seed}
    for arm in ARMS:
        output = root / f"{arm}.json"
        command = build_siam_nb2_proof_run(
            seed_path=str(seed),
            arm=arm,
            output_json=str(output),
            code_revision="siam-estimator-work-test",
            extra_flags=SMOKE_FLAGS,
        )
        assert main(list(command.argv())) == 0
        produced[arm] = output
    return produced


def _payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


@pytest.mark.parametrize("arm", ARMS)
def test_siam_estimator_work_is_exact_for_a_real_short_run(
    siam_smoke_runs: dict[str, Path], arm: str
) -> None:
    record = build_siam_estimator_work_record(
        siam_smoke_runs[arm], seed_path=siam_smoke_runs["seed"], arm=arm
    )

    assert record["schema"] == SIAM_ESTIMATOR_WORK_V1
    assert record["problem_family"] == "siam"
    assert record["parameterization_mode"] == "logical_shared"
    assert record["arm"] == arm
    assert record["segment_role"] == "single_segment"
    assert record["estimator_accounting_route"] == (
        "avqds" if arm == "avqds" else "q_compiled"
    )
    assert record["physical_shots"] is None
    assert record["physical_shots_status"] == PHYSICAL_SHOTS_STATUS

    # SIAM starts from the seed reference with no runtime coordinate, so the
    # first checkpoint is charged g(0) = 2 rather than being refused.
    work = record["estimator_work"]
    assert work["logical_total_estimator_work_status"] == "exact"
    assert work["checkpoint_base_status"] == "exact"
    assert work["candidate_acquisition_status"] == "exact"
    if arm == "avqds":
        assert work["certification_geometry_status"] == "not_applicable"
        assert work["refit_estimator_status"] == "not_applicable"
    else:
        assert work["certification_geometry_status"] == "exact"
        assert work["refit_estimator_status"] == "exact"
    assert work["logical_total_estimator_work"] > 0
    assert work["selection_round_ledger_complete"] is True

    payload = _payload(siam_smoke_runs[arm])
    summary = payload["summary"]
    # p = N_theta is the logical-shared geometry dimension; the terminal Pauli
    # rotation count is carried separately and is larger.
    assert record["n_theta_terminal"] == summary["logical_parameter_count_final"]
    assert record["terminal_pauli_rotation_count"] == (
        summary["runtime_pauli_parameter_count_final"]
    )
    assert record["terminal_pauli_rotation_count"] >= record["n_theta_terminal"]
    assert payload["initial_state"]["runtime_coordinate_labels"] == []
    assert summary["runtime_parameter_count_initial"] == 0


@pytest.mark.parametrize("arm", ARMS)
def test_siam_estimator_work_record_is_digest_bound(
    siam_smoke_runs: dict[str, Path], arm: str, tmp_path: Path
) -> None:
    record = build_siam_estimator_work_record(
        siam_smoke_runs[arm], seed_path=siam_smoke_runs["seed"], arm=arm
    )

    assert verify_siam_estimator_work_record(record) is True

    tampered = copy.deepcopy(record)
    tampered["estimator_work"]["logical_total_estimator_work"] += 1
    assert verify_siam_estimator_work_record(tampered) is False

    destination = tmp_path / f"{arm}_work.json"
    assert write_siam_estimator_work_record(destination, record) == destination
    assert verify_siam_estimator_work_record(_payload(destination)) is True
    # Idempotent rewrite of identical bytes is a no-op.
    write_siam_estimator_work_record(destination, record)
    with pytest.raises(FileExistsError):
        write_siam_estimator_work_record(destination, tampered)


def test_siam_estimator_work_binds_the_seed_and_the_declared_arm(
    siam_smoke_runs: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="structural policy"):
        build_siam_estimator_work_record(
            siam_smoke_runs["exchange"],
            seed_path=siam_smoke_runs["seed"],
            arm="avqds",
        )
    with pytest.raises(ValueError, match="unknown SIAM arm"):
        build_siam_estimator_work_record(
            siam_smoke_runs["exchange"],
            seed_path=siam_smoke_runs["seed"],
            arm="gamma_compiled",
        )

    other_seed = tmp_path / "other_seed.json"
    write_siam_seed_manifest(
        other_seed,
        SIAMScreeningConfig(n_bath=2, u=2.5, label="siam_estimator_work_other"),
    )
    with pytest.raises(ValueError, match="seed digest"):
        build_siam_estimator_work_record(
            siam_smoke_runs["exchange"], seed_path=other_seed, arm="exchange"
        )


def test_siam_estimator_work_refuses_a_bare_continuation_segment(
    siam_smoke_runs: dict[str, Path], tmp_path: Path
) -> None:
    payload = _payload(siam_smoke_runs["exchange"])
    payload["segment_lineage"] = {
        "schema": "siam_raw_segment_lineage_v1",
        "source_run_path": "/nonexistent/screen.json",
        "lineage_sha256": "0" * 64,
    }
    target = _rewrite(tmp_path / "continuation.json", payload)

    with pytest.raises(ValueError, match="bare SIAM continuation segment"):
        build_siam_estimator_work_record(
            target, seed_path=siam_smoke_runs["seed"], arm="exchange"
        )


def test_siam_estimator_work_refuses_a_resumed_segment_marker(
    siam_smoke_runs: dict[str, Path], tmp_path: Path
) -> None:
    payload = _payload(siam_smoke_runs["exchange"])
    payload["trajectory"]["metadata"]["resume_lineage"] = {
        "schema": "siam_raw_segment_lineage_v1",
        "lineage_sha256": "0" * 64,
    }
    target = _rewrite(tmp_path / "resumed.json", payload)

    with pytest.raises(ValueError, match="resumed SIAM segment"):
        build_siam_estimator_work_record(
            target, seed_path=siam_smoke_runs["seed"], arm="exchange"
        )


def test_siam_estimator_work_refuses_a_trajectory_not_starting_at_zero(
    siam_smoke_runs: dict[str, Path], tmp_path: Path
) -> None:
    payload = _payload(siam_smoke_runs["exchange"])
    payload["plot_rows"] = payload["plot_rows"][1:]
    target = _rewrite(tmp_path / "offset.json", payload)

    with pytest.raises(ValueError, match="does not begin"):
        build_siam_estimator_work_record(
            target, seed_path=siam_smoke_runs["seed"], arm="exchange"
        )


def test_siam_estimator_work_accepts_a_stitched_payload(
    siam_smoke_runs: dict[str, Path], tmp_path: Path
) -> None:
    """A stitched payload keeps its last segment's lineage but is accountable."""

    payload = _payload(siam_smoke_runs["exchange"])
    payload["segment_lineage"] = {"schema": "siam_raw_segment_lineage_v1"}
    payload["trajectory"]["metadata"]["resume_lineage"] = {
        "schema": "siam_raw_segment_lineage_v1"
    }
    digest = "a" * 64
    stitch_marker = {
        "schema": SIAM_NB4_STITCH_V1,
        "stitch_sha256": digest,
        "source_run_sha256": ["b" * 64, "c" * 64],
    }
    payload["segment_stitching"] = dict(stitch_marker)
    payload["trajectory"]["metadata"]["segment_stitching"] = dict(stitch_marker)
    target = _rewrite(tmp_path / "stitched.json", payload)

    record = build_siam_estimator_work_record(
        target, seed_path=siam_smoke_runs["seed"], arm="exchange"
    )

    assert record["segment_role"] == "stitched"
    assert record["segment_stitching"]["stitch_sha256"] == digest
    assert record["estimator_work"]["logical_total_estimator_work_status"] == "exact"
    assert verify_siam_estimator_work_record(record) is True


def test_siam_estimator_work_rejects_a_stitch_marker_digest_mismatch(
    siam_smoke_runs: dict[str, Path], tmp_path: Path
) -> None:
    payload = _payload(siam_smoke_runs["exchange"])
    payload["segment_stitching"] = {
        "schema": SIAM_NB4_STITCH_V1,
        "stitch_sha256": "a" * 64,
    }
    payload["trajectory"]["metadata"]["segment_stitching"] = {
        "schema": SIAM_NB4_STITCH_V1,
        "stitch_sha256": "d" * 64,
    }
    target = _rewrite(tmp_path / "bad_stitch.json", payload)

    with pytest.raises(ValueError, match="stitch marker|digest mismatch"):
        build_siam_estimator_work_record(
            target, seed_path=siam_smoke_runs["seed"], arm="exchange"
        )


def test_compiled_q_arm_is_admitted_but_bound_to_its_route_bundle(siam_smoke_runs) -> None:
    from pipelines.time_dynamics.diagnostics import siam_estimator_work as sew

    arm_id, policy, bundle = sew._arm_binding("exchange_q_compiled")
    assert (arm_id, policy, bundle) == (
        "exchange_q_compiled", "exchange", "paper_ii_exchange_q_compiled_v1"
    )
    assert sew._arm_binding("exchange")[2] == "paper_ii_generalized_exchange_v2"
    assert sew._arm_binding("avqds")[2] == sew.ROUTE_NOT_APPLICABLE
    # The historical smoke run's lock records the historical bundle, so the
    # compiled-Q arm label is refused on it: route identity is not inferable
    # from the structural policy alone.
    with pytest.raises(ValueError, match="route bundle"):
        sew.build_siam_estimator_work_record(
            siam_smoke_runs["exchange"], seed_path=siam_smoke_runs["seed"],
            arm="exchange_q_compiled",
        )
    record = sew.build_siam_estimator_work_record(
        siam_smoke_runs["exchange"], seed_path=siam_smoke_runs["seed"], arm="exchange"
    )
    assert record["arm"] == "exchange"
    assert record["exchange_route_bundle"] == "paper_ii_generalized_exchange_v2"
