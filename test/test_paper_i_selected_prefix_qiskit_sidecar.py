from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pipelines.reporting import build_paper_i_selected_prefix_qiskit_sidecar as sidecar


def _serialized(label: str, coefficient: float = 1.0) -> list[dict[str, Any]]:
    return [
        {
            "pauli_exyz": label,
            "coeff_re": coefficient,
            "coeff_im": 0.0,
            "nq": len(label),
        }
    ]


def _beam_payload() -> dict[str, Any]:
    return {
        "adapt_vqe": {
            "history": [
                {"branch_id": 7, "delta_abs_current": 0.2},
                {"branch_id": 9, "delta_abs_current": 0.1},
            ],
            "beam_replay_telemetry": {
                "rounds": [
                    {
                        "frontier": {
                            "branches": [
                                {
                                    "branch_id": 7,
                                    "ansatz_depth": 1,
                                    "operator_labels": ["kept"],
                                }
                            ]
                        }
                    },
                    {
                        "frontier": {
                            "branches": [
                                {
                                    "branch_id": 9,
                                    "ansatz_depth": 2,
                                    "operator_labels": ["kept", "regenerated"],
                                }
                            ]
                        }
                    },
                ]
            },
            "parameterization": {
                "blocks": [
                    {
                        "candidate_label": "kept",
                        "runtime_terms_exyz": _serialized("xy"),
                    }
                ]
            },
        }
    }


def test_beam_prefix_uses_exact_branch_order_and_regenerates_compacted_terms(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    payload = _beam_payload()

    def fake_regeneration(
        _payload: dict[str, Any],
        *,
        source_path: Path,
        labels: list[str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        assert source_path == tmp_path / "result.json"
        assert labels == ["regenerated"]
        return {
            "regenerated": {
                "terms": _serialized("yx", coefficient=-1.0),
                "execution_mode": "grouped_exact",
                "source": "test_regeneration",
            }
        }, {"status": "ok", "regenerated_labels": labels}

    monkeypatch.setattr(sidecar, "_regenerate_missing_term_records", fake_regeneration)
    labels, ops, replay = sidecar.reconstruct_prefix_ansatz(
        payload,
        history_position=2,
        source_path=tmp_path / "result.json",
    )

    assert labels == ["kept", "regenerated"]
    assert [op.label for op in ops] == labels
    assert ops[1].execution_mode == "grouped_exact"
    assert replay["operator_order"]["branch_id"] == 9
    assert replay["replayed_operator_count"] == 2
    assert replay["term_regeneration"]["status"] == "ok"


def test_terminal_beam_prefix_uses_post_frontier_terminal_winner() -> None:
    payload = _beam_payload()
    payload["adapt_vqe"].update(
        {
            "branch_id": 9,
            "ansatz_depth": 1,
            "operators": ["post_terminal_prune"],
        }
    )

    labels, replay = sidecar._beam_prefix_operator_labels(payload, history_position=2)

    assert labels == ["post_terminal_prune"]
    assert replay["source"] == "adapt_vqe.operators_terminal_winner"
    assert replay["ansatz_depth"] == 1


def test_terminal_prefix_uses_authoritative_winner_without_beam_telemetry() -> None:
    payload = {
        "adapt_vqe": {
            "history": [{"branch_id": 4, "delta_abs_current": 0.1}],
            "branch_id": 9,
            "ansatz_depth": 2,
            "operators": ["left", "right"],
        }
    }

    labels, replay = sidecar._beam_prefix_operator_labels(payload, history_position=1)

    assert labels == ["left", "right"]
    assert replay["source"] == "adapt_vqe.operators_terminal_winner"
    assert replay["branch_id"] == 9


def test_record_term_map_recovers_nested_beam_candidate_metadata() -> None:
    payload = {
        "adapt_vqe": {
            "history": [{"selected_records": [{"operator_label": "projected"}]}],
            "continuation": {
                "phase2_scored_rows": [
                    {
                        "candidate_label": "projected",
                        "generator_metadata": {
                            "compile_metadata": {
                                "serialized_terms_exyz": _serialized("xy", coefficient=0.5),
                                "execution_mode": "grouped_exact",
                            }
                        },
                    }
                ]
            },
        }
    }

    term_map = sidecar._record_term_map(payload)

    assert term_map["projected"]["terms"] == _serialized("xy", coefficient=0.5)
    assert term_map["projected"]["execution_mode"] == "grouped_exact"
    assert term_map["projected"]["source"].startswith("payload.")


def test_record_term_map_accepts_global_sign_and_scale_equivalent_metadata() -> None:
    payload = {
        "adapt_vqe": {
            "history": [{"selected_records": []}],
            "parameterization": {
                "blocks": [
                    {
                        "candidate_label": "same_direction",
                        "runtime_terms_exyz": _serialized("xy", coefficient=0.5),
                    }
                ]
            },
            "continuation": {
                "rows": [
                    {
                        "candidate_label": "same_direction",
                        "compile_metadata": {
                            "serialized_terms_exyz": _serialized("xy", coefficient=-2.0)
                        },
                    }
                ]
            },
        }
    }

    term_map = sidecar._record_term_map(payload)

    assert term_map["same_direction"]["terms"] == _serialized("xy", coefficient=0.5)


def test_record_term_map_rejects_conflicting_nested_metadata() -> None:
    payload = {
        "adapt_vqe": {
            "history": [],
            "continuation": {
                "rows": [
                    {
                        "candidate_label": "ambiguous",
                        "compile_metadata": {
                            "serialized_terms_exyz": _serialized("xy", coefficient=0.5)
                        },
                    },
                    {
                        "candidate_label": "ambiguous",
                        "compile_metadata": {
                            "serialized_terms_exyz": _serialized("yx", coefficient=0.5)
                        },
                    },
                ]
            },
        }
    }

    with np.testing.assert_raises_regex(
        ValueError,
        "conflicting coefficient-bearing Pauli metadata",
    ):
        sidecar._record_term_map(payload)


def test_reference_state_accepts_normalized_ansatz_input_state() -> None:
    state, status = sidecar._selected_prefix_reference_state(
        {
            "ansatz_input_state": {
                "nq_total": 2,
                "amplitudes_qn_to_q0": {
                    "01": {"re": 2.0, "im": 0.0},
                },
            }
        },
        num_qubits=2,
    )

    assert status == "statevector_from_ansatz_input_state"
    assert state is not None
    assert np.allclose(state, np.asarray([0.0, 1.0, 0.0, 0.0]))


def test_terminal_winner_query_work_uses_validated_winner_components() -> None:
    payload = {
        "adapt_vqe": {"history": [{}, {}]},
        "summary": {
            "query_work_audit": {
                "status": "ok",
                "S_alg": 15.0,
                "S_alg_work_scope": "winner_lineage_terminal",
                "winner_history_position": 2,
                "components": {
                    "N_H_outer_eval": 1.0,
                    "N_H_refit_eval": 2.0,
                    "N_grad": 3.0,
                    "N_metric": 4.0,
                },
                "N_other_quantum": 5.0,
            }
        },
    }

    work, audit = sidecar._terminal_winner_query_work(payload, history_position=2)

    assert work == {
        "S_alg": 15.0,
        "S_alg_status": "ok",
        "S_alg_work_scope": "winner_lineage_terminal",
        "S_alg_N_H_outer_eval": 1.0,
        "S_alg_N_H_refit_eval": 2.0,
        "S_alg_N_grad_probe": 3.0,
        "S_alg_N_metric_probe": 4.0,
        "S_alg_N_other_quantum": 5.0,
    }
    assert audit["source"] == "summary.query_work_audit"


def test_terminal_winner_query_work_does_not_override_earlier_prefix() -> None:
    payload = {
        "adapt_vqe": {"history": [{}, {}]},
        "summary": {
            "query_work_audit": {
                "status": "ok",
                "S_alg": 1.0,
                "S_alg_work_scope": "winner_lineage_terminal",
                "winner_history_position": 2,
                "components": {
                    "N_H_outer_eval": 1.0,
                    "N_H_refit_eval": 0.0,
                    "N_grad": 0.0,
                    "N_metric": 0.0,
                },
            }
        },
    }

    work, audit = sidecar._terminal_winner_query_work(payload, history_position=1)

    assert work is None
    assert audit["reason"] == "requested_prefix_is_not_terminal"


def test_terminal_winner_query_work_rejects_inconsistent_component_sum() -> None:
    payload = {
        "adapt_vqe": {"history": [{}]},
        "summary": {
            "query_work_audit": {
                "status": "ok",
                "S_alg": 99.0,
                "S_alg_work_scope": "winner_lineage_terminal",
                "winner_history_position": 1,
                "components": {
                    "N_H_outer_eval": 1.0,
                    "N_H_refit_eval": 2.0,
                    "N_grad": 3.0,
                    "N_metric": 4.0,
                },
            }
        },
    }

    work, audit = sidecar._terminal_winner_query_work(payload, history_position=1)

    assert work is None
    assert audit["reason"] == "terminal_query_audit_component_sum_mismatch"


def test_build_sidecar_uses_coefficient_bearing_execution_aware_compiler(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    result_json = tmp_path / "result.json"
    result_json.write_text(
        json.dumps(
            {
                "adapt_vqe": {
                    "history": [
                        {
                            "delta_abs_current": 0.125,
                            "energy_after_opt": -1.0,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "sidecar.json"
    op = sidecar._ansatz_term_from_serialized(
        label="projected::legal_projected",
        terms=[
            {"pauli_exyz": "xy", "coeff_re": 0.5, "coeff_im": 0.0, "nq": 2},
            {"pauli_exyz": "yx", "coeff_re": -0.5, "coeff_im": 0.0, "nq": 2},
        ],
        execution_mode="grouped_exact",
    )
    monkeypatch.setattr(
        sidecar,
        "reconstruct_prefix_ansatz",
        lambda *_args, **_kwargs: ([op.label], [op], {"replayed_operator_count": 1}),
    )
    monkeypatch.setattr(
        sidecar,
        "_selected_prefix_reference_state",
        lambda *_args, **_kwargs: (np.asarray([1.0, 0.0, 0.0, 0.0]), "test"),
    )
    observed: dict[str, Any] = {}

    def fake_compile(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {
            "compiled_resource_qiskit_validated": True,
            "compiled_circuit_stats_status": "ok",
            "compile_convention": sidecar.TABLE_I_QISKIT_COMPILE_CONVENTION,
            "compiled_count_2q_total": 3,
            "compiled_depth_2q_total": 2,
            "compiled_depth_total": 5,
        }

    monkeypatch.setattr(sidecar, "compile_table_i_ansatz_terms", fake_compile)
    monkeypatch.setattr(
        sidecar,
        "snake_algorithmic_work_from_payload",
        lambda *_args, **_kwargs: (
            {
                "S_alg": 11,
                "S_alg_status": "ok",
                "work_scope": "display_prefix",
            },
            {"status": "ok", "scope": "display_prefix"},
        ),
    )
    monkeypatch.setattr(
        sidecar,
        "snake_mechanism_resolved_work_from_payload",
        lambda *_args, **_kwargs: (
            {"mechanism_algorithmic_work": {"S_alg": None, "status": "unavailable"}},
            {"status": "ok"},
        ),
    )
    monkeypatch.setattr(
        sidecar,
        "_sha256_json_without_snake_sidecars",
        lambda _path: "0" * 64,
    )

    payload = sidecar.build_sidecar(
        result_json=result_json,
        history_position=1,
        output_json=output_json,
        threshold=None,
    )

    assert observed["ops"] == [op]
    assert observed["ops"][0].execution_mode == "grouped_exact"
    assert payload["compile_input_semantics"] == (
        "coefficient_bearing_execution_aware_ansatz_terms_v1"
    )
    assert payload["instrumented_runtime_S"] == 11
    assert payload["instrumented_runtime_scope"] == "display_prefix"
    assert payload["paper_i_main_S_convention"] == "paper_i_winning_branch_s_alg_v1"
    assert output_json.is_file()

def test_history_rows_accept_complete_checkpoint_tail() -> None:
    rows = [{"depth": 1}, {"depth": 2}]
    payload = {
        "adapt_vqe": {
            "history_count": 2,
            "history_tail_count": 2,
            "history_tail": rows,
        }
    }

    assert sidecar._history_rows(payload) == rows


def test_build_sidecar_accepts_in_memory_payload_and_streaming_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_json = tmp_path / "large-result.json"
    output_json = tmp_path / "qiskit.json"
    result_json.write_text("not loaded", encoding="utf-8")
    payload = {
        "adapt_vqe": {
            "history": [{"depth": 1}],
            "operators": ["g"],
            "ansatz_depth": 1,
            "abs_delta_e": 1.0e-4,
            "energy": -1.0,
        }
    }
    op = sidecar._ansatz_term_from_serialized(
        label="g",
        terms=[
            {"pauli_exyz": "xx", "coeff_re": 1.0, "coeff_im": 0.0, "nq": 2}
        ],
        execution_mode="grouped_exact",
    )

    monkeypatch.setattr(
        sidecar,
        "_load_json",
        lambda _path: pytest.fail("result JSON must not be reloaded"),
    )
    monkeypatch.setattr(
        sidecar,
        "reconstruct_prefix_ansatz",
        lambda *_args, **_kwargs: (["g"], [op], {"replayed_operator_count": 1}),
    )
    monkeypatch.setattr(
        sidecar,
        "_selected_prefix_reference_state",
        lambda *_args, **_kwargs: (np.asarray([1.0, 0.0, 0.0, 0.0]), "test"),
    )
    monkeypatch.setattr(
        sidecar,
        "compile_table_i_ansatz_terms",
        lambda **_kwargs: {
            "compiled_circuit_stats_status": "ok",
            "compiled_count_2q_total": 1,
            "compiled_depth_2q_total": 1,
            "compiled_depth_total": 1,
        },
    )
    monkeypatch.setattr(
        sidecar,
        "snake_algorithmic_work_from_payload",
        lambda *_args, **_kwargs: ({"S_alg": 1}, {"status": "ok"}),
    )
    monkeypatch.setattr(
        sidecar,
        "snake_mechanism_resolved_work_from_payload",
        lambda *_args, **_kwargs: ({}, {"status": "ok"}),
    )
    monkeypatch.setattr(
        sidecar,
        "_sha256_json_without_snake_sidecars",
        lambda _path: pytest.fail("canonical JSON hash must not reload the file"),
    )

    built = sidecar.build_sidecar(
        result_json=result_json,
        history_position=1,
        output_json=output_json,
        threshold=2.0e-4,
        result_payload=payload,
        source_result_sha256="a" * 64,
        source_result_hash_convention="raw_file_sha256_v1",
    )

    assert built["source_result_sha256"] == "a" * 64
    assert built["source_result_hash_convention"] == "raw_file_sha256_v1"
    assert output_json.is_file()


def test_history_rows_reject_truncated_checkpoint_tail() -> None:
    payload = {
        "adapt_vqe": {
            "history_count": 3,
            "history_tail_count": 2,
            "history_tail": [{"depth": 2}, {"depth": 3}],
        }
    }

    with pytest.raises(ValueError, match="complete"):
        sidecar._history_rows(payload)


def test_history_rows_reconstruct_from_complete_scaffold_when_tail_is_compact() -> None:
    scaffold = [
        {
            "step_index": 1,
            "energy_after_opt": -1.0,
            "selected_records": [{"generator_label": "xe", "position_id": 0}],
        },
        {
            "step_index": 2,
            "energy_after_opt": -1.1,
            "selected_records": [{"generator_label": "ey", "position_id": 1}],
        },
        {
            "step_index": 3,
            "energy_after_opt": -1.2,
            "selected_records": [{"generator_label": "ze", "position_id": 2}],
        },
    ]
    payload = {
        "adapt_vqe": {
            "history_count": 3,
            "history_tail_count": 1,
            "history": [{"depth": 3}],
            "history_tail": [{"depth": 3}],
            "continuation": {"selected_scaffold_history": scaffold},
        }
    }

    rows = sidecar._history_rows(payload)

    assert [row["depth"] for row in rows] == [1, 2, 3]
    assert rows[1]["selected_records"][0]["generator_label"] == "ey"
