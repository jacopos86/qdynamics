from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pipelines.scaffold import hh_continuation_scoring as scoring


@pytest.mark.parametrize(
    ("selector_name", "proposal_builder_name"),
    [
        (
            "greedy_reduced_plane_batch_select",
            "greedy_reduced_plane_batch_proposals",
        ),
        (
            "combinatorial_reduced_plane_batch_select",
            "combinatorial_reduced_plane_batch_proposals",
        ),
    ],
)
def test_batch_selector_preserves_complete_geometry_workspace_receipt(
    monkeypatch: pytest.MonkeyPatch,
    selector_name: str,
    proposal_builder_name: str,
) -> None:
    selected_records = (
        {"candidate_label": "candidate-a", "full_v2_score": 3.0},
        {"candidate_label": "candidate-b", "full_v2_score": 2.0},
    )
    proposal_summary = {
        "selection_mode": "proposal_response_marker",
        "joint_gain": 5.0,
        "G": [[1.0, 0.0], [0.0, 1.0]],
    }
    proposal = scoring.BatchSelectionProposal(
        records=selected_records,
        summary=proposal_summary,
        score=5.0,
        delta_e3=5.0,
        k3=0.0,
        denominator_1_plus_k3=1.0,
    )
    complete_summary = {
        **proposal_summary,
        "geometry_workspace": {
            "schema": "batch_full_geometry_workspace_v1",
            "active_indices": [0, 2],
        },
        "proposal_count": 1,
    }

    def _fake_proposal_builder(*_args: Any, **_kwargs: Any):
        return [proposal], complete_summary

    monkeypatch.setattr(scoring, proposal_builder_name, _fake_proposal_builder)
    selector: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] = getattr(
        scoring,
        selector_name,
    )
    records, summary = selector(
        [],
        cfg=object(),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.ones(1, dtype=complex),
        psi_state=np.ones(1, dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
    )

    assert records == [dict(record) for record in selected_records]
    assert summary == complete_summary
    assert summary["geometry_workspace"]["active_indices"] == [0, 2]
    assert summary is not complete_summary


@pytest.mark.parametrize(
    ("selector_name", "proposal_builder_name"),
    [
        (
            "greedy_reduced_plane_batch_select",
            "greedy_reduced_plane_batch_proposals",
        ),
        (
            "combinatorial_reduced_plane_batch_select",
            "combinatorial_reduced_plane_batch_proposals",
        ),
    ],
)
def test_batch_selector_preserves_empty_proposal_summary(
    monkeypatch: pytest.MonkeyPatch,
    selector_name: str,
    proposal_builder_name: str,
) -> None:
    complete_summary = {
        "selection_mode": "no_feasible_proposal",
        "geometry_workspace": {"active_indices": []},
    }

    def _fake_proposal_builder(*_args: Any, **_kwargs: Any):
        return [], complete_summary

    monkeypatch.setattr(scoring, proposal_builder_name, _fake_proposal_builder)
    selector = getattr(scoring, selector_name)
    records, summary = selector(
        [],
        cfg=object(),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=np.ones(1, dtype=complex),
        psi_state=np.ones(1, dtype=complex),
        h_compiled=object(),
        novelty_oracle=object(),
        curvature_oracle=object(),
    )

    assert records == []
    assert summary == complete_summary
    assert summary is not complete_summary
