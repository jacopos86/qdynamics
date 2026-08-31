"""Frozen expansion contract for the Paper-II parallel benchmark matrix."""

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from pipelines.time_dynamics.parallel_benchmark_matrix import (
    DEFAULT_MATRIX_PATH,
    ParallelBenchmarkMatrix,
)


def test_authorized_matrix_expands_to_exact_declared_cells() -> None:
    matrix = ParallelBenchmarkMatrix.read()
    hh = matrix.cells_for("paper_ii_hh_product_v1")
    siam = matrix.cells_for("siam_nb4_pilot_v1")
    hubbard = matrix.cells_for("paper_ii_hubbard_l4_scale_v1")
    assert len(hh) == 36
    assert len(siam) == 1
    assert len(hubbard) == 2
    assert len(matrix.cells) == 39
    assert all(cell.controller == "state_motion_5e-5" for cell in matrix.cells)
    assert {cell.physics["n_ph_max"] for cell in hh} == {1, 3}
    assert {cell.physics["regime"] for cell in hh} == {
        "mott_dominated",
        "crossover",
        "phonon_dominated",
    }
    assert {cell.drive for cell in hubbard} == {"weakslow", "strongfast"}


def test_matrix_preserves_cost_and_matching_decisions() -> None:
    matrix = ParallelBenchmarkMatrix.read()
    common = matrix.payload["common"]
    policy = matrix.payload["accuracy_policy"]
    assert common["qiskit_compilation"] is False
    assert common["cost_authority"] == (
        "logical_n_theta_and_routing_independent_graph_tuple"
    )
    assert policy["aggregate"] == "rms_of_normalized_q_components"
    assert policy["component_deadband"] == 0.1
    assert policy["componentwise_match_required"] is True
    assert policy["matched_factor"] == 2.0
    assert policy["response_scale_floor"] == 1.0e-12
    assert policy["absolute_guards_by_lane"] == {
        "paper_ii_hh_product_v1": {
            "mean_abs_energy_error": 1.0e-2,
            "mean_abs_doublon_error": 5.0e-3,
            "mean_max_abs_site_occupation_error": 5.0e-3,
        },
        "paper_ii_hubbard_l4_scale_v1": {
            "mean_abs_energy_error": 1.0e-2,
            "mean_abs_doublon_error": 5.0e-3,
            "mean_max_abs_site_occupation_error": 5.0e-3,
        },
    }


def test_matrix_builds_complete_policy_from_verified_reference_scales() -> None:
    matrix = ParallelBenchmarkMatrix.read()
    scales = {
        "weakslow": {
            horizon: {
                "energy_rms_excursion": 0.2,
                "doublon_rms_excursion": 0.1,
                "max_site_occupation_rms_excursion": 0.1,
            }
            for horizon in ("t5", "t20")
        }
    }
    policy = matrix.canonical_accuracy_policy(
        "paper_ii_hh_product_v1", scales
    )
    assert policy.response_scales == scales
    assert policy.response_relative_target == 0.05
    assert policy.response_relative_guard == 0.15
    assert policy.infidelity_ceiling == 1.0e-4
    assert policy.matched_accuracy_factor == 2.0
    assert policy.matched_component_deadband == 0.10
    assert policy.response_scale_floor == 1.0e-12
    assert policy.absolute_energy_error_ceiling == 1.0e-2
    assert policy.absolute_doublon_error_ceiling == 5.0e-3
    assert policy.absolute_site_occupation_error_ceiling == 5.0e-3
    assert matrix.accuracy_policy_spec_sha256 == hashlib.sha256(
        json.dumps(
            matrix.payload["accuracy_policy"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def test_matrix_rejects_accuracy_policy_drift(tmp_path) -> None:
    matrix = ParallelBenchmarkMatrix.read()
    payload = copy.deepcopy(matrix.payload)
    payload["accuracy_policy"]["observable_relative_guard"] = 0.50
    changed_path = tmp_path / "changed-policy.json"
    changed_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="complete accuracy policy"):
        ParallelBenchmarkMatrix.read(changed_path)


@pytest.mark.parametrize(
    ("label", "mutation"),
    (
        (
            "handoff",
            lambda payload: payload["decision_handoff"].__setitem__(
                "sha256", "0" * 64
            ),
        ),
        (
            "cost authority",
            lambda payload: payload["common"].__setitem__(
                "cost_authority", "compiled_qiskit_depth"
            ),
        ),
        (
            "Qiskit exclusion",
            lambda payload: payload["common"].__setitem__(
                "qiskit_compilation", True
            ),
        ),
        (
            "drive waveform",
            lambda payload: payload["common"]["drive_waveform"].__setitem__(
                "operator", "uniform_site_density"
            ),
        ),
        (
            "ordered grid",
            lambda payload: payload["common"]["driven_grid"].__setitem__(
                "points", 502
            ),
        ),
        (
            "activation ladder",
            lambda payload: payload["activation_ladders"][
                "hh_and_hubbard_l4"
            ].__setitem__("floor", 1.0e-11),
        ),
        (
            "excluded scope",
            lambda payload: payload["excluded"].append("future_forbidden_lane"),
        ),
        (
            "seed count",
            lambda payload: payload["hh_product"].__setitem__(
                "independent_seed_count", 5
            ),
        ),
        (
            "reference count",
            lambda payload: payload["hubbard_l4"].__setitem__(
                "exact_reference_count", 1
            ),
        ),
    ),
)
def test_matrix_rejects_every_frozen_decision_drift(
    tmp_path, label, mutation
) -> None:
    payload = copy.deepcopy(ParallelBenchmarkMatrix.read().payload)
    mutation(payload)
    changed_path = tmp_path / f"changed-{label.replace(' ', '-')}.json"
    changed_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen matrix payload"):
        ParallelBenchmarkMatrix.read(changed_path)


def test_matrix_rejects_self_consistent_but_incomplete_cell_product(tmp_path) -> None:
    payload = copy.deepcopy(ParallelBenchmarkMatrix.read().payload)
    payload["hh_product"]["drives"] = payload["hh_product"]["drives"][:-1]
    payload["hh_product"]["physical_cell_count"] = 30
    payload["hh_product"]["method_slot_count"] = 60
    payload["hh_product"]["exact_reference_count"] = 30
    changed_path = tmp_path / "thirty-three-cells.json"
    changed_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen matrix payload|exactly 39"):
        ParallelBenchmarkMatrix.read(changed_path)


def test_matrix_carries_complete_payload_and_exact_file_digests(tmp_path) -> None:
    matrix = ParallelBenchmarkMatrix.read()
    assert matrix.digest == matrix.payload_sha256
    assert matrix.payload_sha256 == hashlib.sha256(
        json.dumps(
            matrix.payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    assert matrix.raw_sha256 == hashlib.sha256(
        DEFAULT_MATRIX_PATH.read_bytes()
    ).hexdigest()

    reformatted = tmp_path / "matrix.json"
    reformatted.write_text(json.dumps(matrix.payload, indent=7), encoding="utf-8")
    reread = ParallelBenchmarkMatrix.read(reformatted)
    assert reread.payload_sha256 == matrix.payload_sha256
    assert reread.raw_sha256 != matrix.raw_sha256


def test_expanded_semantic_digest_binds_every_execution_decision() -> None:
    matrix = ParallelBenchmarkMatrix.read()
    payload = copy.deepcopy(matrix.payload)
    payload["common"]["drive_waveform"]["operator"] = "uniform_site_density"
    payload["common"]["cost_authority"] = "compiled_qiskit_depth"
    payload["common"]["qiskit_compilation"] = True
    changed = replace(matrix, payload=payload)
    assert changed.expanded_semantic_sha256 != matrix.expanded_semantic_sha256
    assert changed.payload_sha256 != matrix.payload_sha256
