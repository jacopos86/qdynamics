from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pipelines.exact_bench.paper_i_hh_recovery_prefix_qiskit_sidecar import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_ORDER_REPAIR_SCHEMA,
    CURRENT_JR_BACKEND,
    CURRENT_JR_CONVENTION,
    HISTORICAL_DISPLAYED_CONVENTION,
    SIDECAR_SCHEMA,
    _canonical_sha256,
    build_sidecar,
    build_checkpoint_order_repair_record,
    derive_execution_order_repaired_checkpoint,
    resolve_active_prefix_checkpoint,
    validate_active_prefix_checkpoint,
)


def _checkpoint(
    *,
    outer_iteration: int,
    labels_and_paulis: tuple[tuple[str, str, float], ...],
) -> dict:
    blocks = []
    operators = []
    runtime_parameters = []
    logical_parameters = []
    for logical_index, (label, pauli, theta) in enumerate(labels_and_paulis):
        term = {
            "pauli_exyz": pauli,
            "coeff_re": 1.0,
            "coeff_im": 0.0,
            "nq": len(pauli),
        }
        blocks.append(
            {
                "candidate_label": label,
                "logical_index": logical_index,
                "runtime_start": logical_index,
                "runtime_count": 1,
                "runtime_terms_exyz": [dict(term)],
            }
        )
        operators.append(
            {
                "active_position": logical_index,
                "label": label,
                "generator_id": f"gen:{logical_index}",
                "parent_generator_id": f"parent:{logical_index}",
                "execution_mode": "termwise_product",
                "serialized_terms_exyz_in_execution_order": [dict(term)],
                "runtime_split": {"mode": "archival_child_set_forward_v1"},
                "route_a_child_padding_lineage": {"policy": "exact_projected_grouped_v1"},
            }
        )
        runtime_parameters.append(theta)
        logical_parameters.append(theta)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_kind": "post_admission_prune",
        "outer_iteration": outer_iteration,
        "active_ansatz_depth": len(labels_and_paulis),
        "ordered_active_operator_labels": [row[0] for row in labels_and_paulis],
        "ordered_active_operators": operators,
        "signed_unwrapped_runtime_parameters": runtime_parameters,
        "signed_unwrapped_logical_parameters": logical_parameters,
        "parameterization_mode": "logical_shared",
        "parameterization": {
            "mode": "per_pauli_term_v1",
            "term_order": "sorted",
            "ignore_identity": True,
            "coefficient_tolerance": 1.0e-12,
            "logical_operator_count": len(blocks),
            "runtime_parameter_count": len(blocks),
            "blocks": blocks,
        },
        "projective_state_fingerprint": f"state:{outer_iteration}",
        "fixed_spin_sector_probability": 1.0,
        "fixed_spin_sector_illegal_probability": 0.0,
        "boson_legal_codeword_probability": 1.0,
        "boson_illegal_codeword_probability": 0.0,
        "boson_legal_subspace": {"policy": "binary_padding"},
        "admission_at_outer_iteration": {
            "selected_batch_labels": [labels_and_paulis[-1][0]],
            "selected_batch_positions": [len(labels_and_paulis) - 1],
            "selected_batch_effective_positions": [len(labels_and_paulis) - 1],
        },
        "post_admission_prune": {"executed": True},
    }
    checkpoint["checkpoint_sha256"] = _canonical_sha256(checkpoint)
    return checkpoint


def _result_payload(checkpoints: list[dict]) -> dict:
    return {
        "ansatz_input_state": {
            "nq_total": 2,
            "source": "hf",
            "handoff_state_kind": "reference_state",
            "amplitudes_qn_to_q0": {"01": {"re": 1.0, "im": 0.0}},
        },
        "adapt_vqe": {
            "continuation": {
                "active_prefix_checkpoints": checkpoints,
            }
        },
    }


def _permuted_multiterm_checkpoint(*, substantive_mismatch: bool = False) -> dict:
    checkpoint = _checkpoint(
        outer_iteration=7,
        labels_and_paulis=(("child", "xx", -0.25),),
    )
    first = {
        "pauli_exyz": "xx",
        "coeff_re": 0.5,
        "coeff_im": 0.0,
        "nq": 2,
    }
    second = {
        "pauli_exyz": "zz" if not substantive_mismatch else "yy",
        "coeff_re": 0.75,
        "coeff_im": 0.0,
        "nq": 2,
    }
    expected_second = dict(second)
    if substantive_mismatch:
        expected_second["pauli_exyz"] = "zz"
    checkpoint["parameterization"]["blocks"][0]["runtime_count"] = 2
    checkpoint["parameterization"]["blocks"][0]["runtime_terms_exyz"] = [
        dict(first),
        expected_second,
    ]
    checkpoint["parameterization"]["runtime_parameter_count"] = 2
    checkpoint["signed_unwrapped_runtime_parameters"] = [-0.25, -0.25]
    checkpoint["ordered_active_operators"][0][
        "serialized_terms_exyz_in_execution_order"
    ] = [dict(second), dict(first)]
    checkpoint.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = _canonical_sha256(checkpoint)
    return checkpoint


def test_resolve_checkpoint_uses_explicit_post_prune_active_order() -> None:
    before_prune = _checkpoint(
        outer_iteration=4,
        labels_and_paulis=(("a", "xx", 0.2), ("b", "zz", -0.3)),
    )
    after_prune = _checkpoint(
        outer_iteration=5,
        labels_and_paulis=(("b", "zz", -0.35),),
    )
    payload = _result_payload([before_prune, after_prune])

    resolution = resolve_active_prefix_checkpoint(payload, outer_iteration=5)
    validated = validate_active_prefix_checkpoint(
        resolution.checkpoint, expected_outer_iteration=5
    )

    assert validated.checkpoint["ordered_active_operator_labels"] == ["b"]
    assert validated.layout.logical_parameter_count == 1
    assert validated.runtime_parameters.tolist() == [-0.35]
    assert resolution.locations == (
        "adapt_vqe.continuation.active_prefix_checkpoints[1]",
    )


def test_checkpoint_hash_mismatch_fails_closed() -> None:
    checkpoint = _checkpoint(
        outer_iteration=3,
        labels_and_paulis=(("a", "xz", 0.25),),
    )
    checkpoint["signed_unwrapped_runtime_parameters"] = [0.5]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_active_prefix_checkpoint(checkpoint, expected_outer_iteration=3)


def test_execution_order_repair_is_explicit_and_permutation_only() -> None:
    checkpoint = _permuted_multiterm_checkpoint()
    with pytest.raises(ValueError, match="execution terms disagree"):
        validate_active_prefix_checkpoint(checkpoint, expected_outer_iteration=7)

    repaired, repair = derive_execution_order_repaired_checkpoint(
        checkpoint,
        expected_outer_iteration=7,
    )
    validated = validate_active_prefix_checkpoint(
        repaired,
        expected_outer_iteration=7,
    )

    assert repair["status"] == "repaired_permutation_only"
    assert repair["repaired_operator_indices"] == [0]
    assert repair["source_checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    assert repair["repaired_checkpoint_sha256"] == validated.checkpoint_sha256
    assert repair["substantive_term_changes"] is False
    assert repaired["ordered_active_operators"][0][
        "serialized_terms_exyz_in_execution_order"
    ] == repaired["parameterization"]["blocks"][0]["runtime_terms_exyz"]


def test_execution_order_repair_rejects_substantive_term_drift() -> None:
    checkpoint = _permuted_multiterm_checkpoint(substantive_mismatch=True)
    with pytest.raises(ValueError, match="differ substantively"):
        derive_execution_order_repaired_checkpoint(
            checkpoint,
            expected_outer_iteration=7,
        )


def test_checkpoint_order_repair_record_preserves_source_lock(tmp_path: Path) -> None:
    checkpoint = _permuted_multiterm_checkpoint()
    result_json = tmp_path / "result.json"
    result_json.write_text(
        json.dumps(_result_payload([checkpoint]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    record = build_checkpoint_order_repair_record(
        result_json=result_json,
        outer_iteration=7,
        expected_checkpoint_sha256=checkpoint["checkpoint_sha256"],
    )

    assert record["schema"] == CHECKPOINT_ORDER_REPAIR_SCHEMA
    assert record["source"]["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    assert record["repair"]["repaired_operator_count"] == 1
    assert record["repaired_checkpoint"]["checkpoint_sha256"] == record["repair"][
        "repaired_checkpoint_sha256"
    ]


def test_nonidentical_duplicate_iteration_fails_closed() -> None:
    left = _checkpoint(
        outer_iteration=3,
        labels_and_paulis=(("a", "xz", 0.25),),
    )
    right = _checkpoint(
        outer_iteration=3,
        labels_and_paulis=(("b", "yz", 0.25),),
    )
    payload = _result_payload([left, right])
    with pytest.raises(ValueError, match="Multiple nonidentical"):
        resolve_active_prefix_checkpoint(payload, outer_iteration=3)


def test_build_sidecar_keeps_historical_and_jr_conventions_separate(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(
        outer_iteration=3,
        labels_and_paulis=(("child", "xz", -0.375),),
    )
    result_json = tmp_path / "result.json"
    result_json.write_text(
        json.dumps(_result_payload([checkpoint]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sidecar = build_sidecar(result_json=result_json, outer_iteration=3)

    assert sidecar["schema"] == SIDECAR_SCHEMA
    assert sidecar["source"]["checkpoint_hash_verified"] is True
    assert sidecar["prefix"]["prune_aware"] is True
    assert sidecar["prefix"]["ordered_active_operator_labels"] == ["child"]
    historical = sidecar["historical_displayed_convention"]
    current_jr = sidecar["current_jr_fake_marrakesh_convention"]
    assert historical["identity"] == HISTORICAL_DISPLAYED_CONVENTION
    assert historical["backend"] is None
    assert historical["optimization_level"] == 0
    assert historical["seed_transpiler"] == 7
    assert current_jr["identity"] == CURRENT_JR_CONVENTION
    assert current_jr["requested_backend"] == CURRENT_JR_BACKEND
    assert current_jr["resolved_backend"] == CURRENT_JR_BACKEND
    assert current_jr["optimization_level"] == 1
    assert current_jr["seed_transpiler"] == 7
    for compile_block in (historical, current_jr):
        assert set(compile_block["metrics"]) == {"N2q", "D2q", "Dc"}
        assert compile_block["metrics"]["N2q"] >= 0
        assert compile_block["metrics"]["D2q"] >= 0
        assert compile_block["metrics"]["Dc"] >= compile_block["metrics"]["D2q"]
    assert sidecar["convention_comparison"]["same_convention"] is False


def test_expected_source_locks_fail_closed(tmp_path: Path) -> None:
    checkpoint = _checkpoint(
        outer_iteration=2,
        labels_and_paulis=(("child", "xz", 0.1),),
    )
    result_json = tmp_path / "result.json"
    result_json.write_text(json.dumps(_result_payload([checkpoint])), encoding="utf-8")
    with pytest.raises(ValueError, match="Result SHA-256 mismatch"):
        build_sidecar(
            result_json=result_json,
            outer_iteration=2,
            expected_result_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="caller lock"):
        build_sidecar(
            result_json=result_json,
            outer_iteration=2,
            expected_checkpoint_sha256="0" * 64,
        )
