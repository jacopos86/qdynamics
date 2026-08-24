from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.quantum.ansatz_parameterization import build_parameter_layout, runtime_insert_position
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt import batch_ordering


def _poly(pauli: str) -> PauliPolynomial:
    return PauliPolynomial("JW", [PauliTerm(len(pauli), ps=pauli, pc=1.0)])


def _term(label: str, pauli: str = "x") -> AnsatzTerm:
    return AnsatzTerm(label=str(label), polynomial=_poly(pauli))


def _feature(**overrides: object) -> CandidateFeatures:
    base = dict(
        stage_name="phase3",
        candidate_label="a",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        g_signed=0.25,
        g_abs=0.25,
        g_lcb=0.25,
        sigma_hat=0.0,
        F=1.0,
        novelty=0.8,
        curvature_mode="current_curv",
        novelty_mode="current_novelty",
        refit_window_indices=[0],
        compiled_position_cost_proxy={},
        measurement_cache_stats={},
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        simple_score=0.25,
        score_version="test_v1",
        phase2_raw_score=0.25,
        full_v2_score=0.25,
        selector_score=0.25,
        phase_score_components={"existing": 1.0},
        actual_fallback_mode="exact_reduced",
    )
    base.update(overrides)
    return CandidateFeatures(**base)


def _record(label: str, idx: int, *, score: float = 1.0, position: int = 0) -> dict[str, Any]:
    term = _term(label, "x" if idx % 2 == 0 else "y")
    feat = _feature(
        candidate_label=label,
        candidate_pool_index=idx,
        position_id=position,
        append_position=position,
        positions_considered=[position],
        simple_score=score,
        phase2_raw_score=score,
        full_v2_score=score,
        selector_score=score,
    )
    return {
        "candidate_label": label,
        "candidate_pool_index": idx,
        "position_id": position,
        "candidate_term": term,
        "feature": feat,
        "simple_score": score,
        "phase2_raw_score": score,
        "full_v2_score": score,
    }


def _layout(ops: list[AnsatzTerm]):
    return build_parameter_layout(
        ops,
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )


def _splice_candidate_at_position(
    *,
    ops: list[AnsatzTerm],
    theta: np.ndarray,
    op: AnsatzTerm,
    position_id: int,
    init_theta: float = 0.0,
) -> tuple[list[AnsatzTerm], np.ndarray]:
    current_layout = _layout(ops)
    op_layout = _layout([op])
    pos_logical = max(0, min(int(current_layout.logical_parameter_count), int(position_id)))
    pos_runtime = int(runtime_insert_position(current_layout, pos_logical))
    new_ops = list(ops)
    new_ops.insert(pos_logical, op)
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    insert_block = np.full(int(op_layout.runtime_parameter_count), float(init_theta), dtype=float)
    return new_ops, np.asarray(np.insert(theta_arr, pos_runtime, insert_block), dtype=float)


def _config(**overrides: object) -> batch_ordering.BatchOrderingConfig:
    values = dict(
        mode="finite_step_v1",
        max_permutations=8,
        rho=0.2,
        batch_target_size=3,
        batch_size_cap=3,
        batch_near_degenerate_ratio=0.8,
    )
    values.update(overrides)
    return batch_ordering.BatchOrderingConfig(**values)


def _runtime(pool: list[AnsatzTerm], *, preferred_order: tuple[str, ...] = ()) -> batch_ordering.BatchOrderingRuntime:
    def evaluate_selected_energy_objective(**kwargs: Any) -> float:
        labels = tuple(str(op.label) for op in kwargs["ops_now"])
        suffix = labels[-len(preferred_order):] if preferred_order else ()
        return 0.0 if suffix == preferred_order else 1.0

    return batch_ordering.BatchOrderingRuntime(
        pool=pool,
        adapt_state_backend_key="legacy",
        build_selected_layout=_layout,
        build_compiled_executor=lambda ops: object(),
        splice_candidate_at_position=_splice_candidate_at_position,
        evaluate_selected_energy_objective=evaluate_selected_energy_objective,
    )


def test_score_sorted_and_singleton_paths_preserve_records_without_proxy() -> None:
    records = [_record("a", 0), _record("b", 1)]
    ordered, summary = batch_ordering._order_batch_records_for_admission(
        records=records,
        base_ops=[],
        base_theta=np.zeros(0),
        base_layout=_layout([]),
        depth_one_based=2,
        config=_config(mode="score_sorted"),
        runtime=_runtime([rec["candidate_term"] for rec in records]),
    )

    assert [row["candidate_label"] for row in ordered] == ["a", "b"]
    assert summary["reason"] == "score_sorted_legacy"
    assert "batch_order_proxy" not in ordered[0]

    singleton, singleton_summary = batch_ordering._order_batch_records_for_admission(
        records=[records[0]],
        base_ops=[],
        base_theta=np.zeros(0),
        base_layout=_layout([]),
        depth_one_based=3,
        config=_config(),
        runtime=_runtime([rec["candidate_term"] for rec in records]),
    )
    assert singleton == [records[0]]
    assert singleton_summary["reason"] == "singleton_batch"


def test_finite_step_ordering_scores_permutations_and_does_not_mutate_inputs() -> None:
    records = [_record("a", 0), _record("b", 1)]
    original_first = dict(records[0])
    ordered, summary = batch_ordering._order_batch_records_for_admission(
        records=records,
        base_ops=[],
        base_theta=np.zeros(0),
        base_layout=_layout([]),
        depth_one_based=1,
        config=_config(max_permutations=8),
        runtime=_runtime([rec["candidate_term"] for rec in records], preferred_order=("b", "a")),
    )

    assert [row["candidate_label"] for row in ordered] == ["b", "a"]
    assert summary["reason"] == "finite_step_proxy_scored"
    assert summary["selected"] is True
    assert summary["reordered"] is True
    assert summary["best_energy_proxy"] == 0.0
    assert ordered[0]["batch_order_proxy"]["rank"] == 0
    assert ordered[1]["batch_order_proxy"]["rank"] == 1
    assert records[0] == original_first
    assert "batch_order_proxy" not in records[0]


def test_invalid_finite_step_records_return_no_valid_ordering_summary() -> None:
    ordered, summary = batch_ordering._order_batch_records_for_admission(
        records=[{"candidate_label": "bad"}],
        base_ops=[],
        base_theta=np.zeros(0),
        base_layout=_layout([]),
        depth_one_based=1,
        config=_config(),
        runtime=_runtime([]),
        score_singleton=True,
    )

    assert ordered[0]["candidate_label"] == "bad"
    assert ordered[0]["batch_order_proxy"]["best_energy_proxy"] is None
    assert summary["reason"] == "finite_step_proxy_scored"
    assert summary["selected"] is True
    assert summary["orders_scored_sample"][0]["valid"] is False
    assert summary["orders_scored_sample"][0]["reason"] == "invalid_record"


def test_finite_step_order_rescue_fills_positive_then_dormant_candidates() -> None:
    selected = [_record("a", 0, score=10.0)]
    source = [
        selected[0],
        _record("b", 1, score=9.0),
        _record("c", 2, score=2.0),
        _record("d", 3, score=0.0),
        _record("ham_full", 4, score=0.0),
    ]

    def sort_key(row: Mapping[str, Any]) -> tuple[float, int]:
        return (-float(row.get("full_v2_score", 0.0)), int(row.get("candidate_pool_index", -1)))

    chosen, summary = batch_ordering._finite_step_order_rescue_records(
        source_records=source,
        selected_records=selected,
        depth_one_based=5,
        config=_config(batch_target_size=4, batch_size_cap=4, batch_near_degenerate_ratio=0.8),
        record_sort_key=sort_key,
    )

    assert [row["candidate_label"] for row in chosen] == ["a", "b", "c", "d"]
    assert chosen[1]["batch_order_rescue"]["source"] == "near_degenerate_shell"
    assert chosen[2]["batch_order_rescue"]["source"] == "positive_score_fill"
    assert chosen[3]["batch_order_rescue"]["source"] == "dormant_finite_step_fill"
    assert chosen[3]["batch_order_rescue"]["dormant"] is True
    assert summary["used"] is True
    assert summary["reason"] == "finite_step_positive_dormant_shell"
    assert summary["dormant_candidate_count"] == 2


def test_finite_step_order_rescue_blocks_alternate_position_of_same_pauli_child() -> None:
    selected = [
        {
            **_record("parent-a::child-x", 0, score=10.0),
            "route_a_global_pauli_identity": "pauli:x",
        }
    ]
    alternate_position = {
        **_record("parent-b::child-x", 1, score=9.5, position=1),
        "route_a_global_pauli_identity": "pauli:x",
    }
    sibling = {
        **_record("parent-a::child-y", 2, score=9.0),
        "route_a_global_pauli_identity": "pauli:y",
    }

    chosen, _summary = batch_ordering._finite_step_order_rescue_records(
        source_records=[selected[0], alternate_position, sibling],
        selected_records=selected,
        depth_one_based=5,
        config=_config(
            batch_target_size=2,
            batch_size_cap=2,
            batch_near_degenerate_ratio=0.8,
        ),
        record_sort_key=lambda row: (
            -float(row.get("full_v2_score", 0.0)),
            int(row.get("candidate_pool_index", -1)),
        ),
    )

    assert [row["candidate_label"] for row in chosen] == [
        "parent-a::child-x",
        "parent-a::child-y",
    ]


def test_schur_batch_context_serializes_keyed_maps_and_malformed_inputs() -> None:
    records = [
        {"candidate_pool_index": 0, "position_id": 0, "candidate_label": "a"},
        {"candidate_pool_index": 1, "position_id": 1, "candidate_label": "b"},
    ]
    payload = batch_ordering._schur_batch_context_from_summary(
        records=records,
        batch_summary={
            "alpha": [0.1, 0.2],
            "G": [[1.0, 0.0], [0.0, 1.0]],
            "common_window_indices": ["0", "1"],
            "schur_window_solves": [[1.0, 2.0], [3.0, 4.0]],
            "joint_gain": 0.5,
        },
    )

    assert payload["schema"] == "static_adapt_batch_schur_context_v1"
    assert payload["record_keys"] == [[0, 0, "a"], [1, 1, "b"]]
    assert payload["common_window_indices"] == [0, 1]
    assert payload["alpha_abs_by_key"]['[0, 0, "a"]'] == 0.1
    assert payload["schur_window_solve_by_key"]['[1, 1, "b"]'] == [3.0, 4.0]
    assert payload["joint_gain"] == 0.5

    assert (
        batch_ordering._schur_batch_context_from_summary(
            records=records,
            batch_summary={"alpha": [0.1], "schur_window_solves": [[1.0]]},
        )
        == {}
    )
