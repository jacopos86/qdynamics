from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    CANDIDATE_REPRESENTATION_MACRO,
)
from pipelines.static_adapt.ra_adapt.pools import (
    build_executable_macro_pool,
    build_guarded_single_pauli_pool,
    build_parent_template_inventory,
)


_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_fixture(n_ph_max: int) -> dict[str, Any]:
    path = _FIXTURE_DIR / f"ra_adapt_pool_inventory_{n_ph_max}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "ra_adapt_pool_inventory_fixture_v1"
    return payload


@pytest.mark.parametrize(
    ("n_ph_max", "parent_count", "removed_count", "macro_count"),
    ((3, 123, 21, 102), (7, 171, 23, 148)),
)
def test_resolved_pool_inventory_matches_lock_fixture(
    n_ph_max: int,
    parent_count: int,
    removed_count: int,
    macro_count: int,
) -> None:
    fixture = _load_fixture(n_ph_max)
    problem = resolve_problem_context(
        ProblemRequest(**dict(fixture["problem"]))
    )
    parent = build_parent_template_inventory(
        problem,
        representation_id=CANDIDATE_REPRESENTATION_MACRO,
    ).receipt
    macro = build_executable_macro_pool(problem).receipt
    guarded = build_guarded_single_pauli_pool(problem).receipt

    expected_parent = fixture["parent_inventory"]
    assert parent.count == parent_count
    assert list(parent.ordered_labels) == expected_parent["ordered_labels"]
    assert (
        parent.ordered_labels_sha256
        == expected_parent["ordered_labels_sha256"]
    )
    assert (
        parent.ordered_pool_sha256
        == expected_parent["ordered_pool_sha256"]
    )

    expected_macro = fixture["executable_macro_pool"]
    assert macro.count == macro_count
    assert len(macro.removed_labels) == removed_count
    assert list(macro.removed_labels) == expected_macro["removed_labels"]
    assert list(macro.ordered_labels) == expected_macro["ordered_labels"]
    assert (
        macro.ordered_labels_sha256
        == expected_macro["ordered_labels_sha256"]
    )
    assert (
        macro.ordered_pool_sha256
        == expected_macro["ordered_pool_sha256"]
    )
    assert (
        macro.source_parent_ordered_labels_sha256
        == expected_macro["source_parent_ordered_labels_sha256"]
    )
    removed = set(macro.removed_labels)
    assert [
        label
        for label in parent.ordered_labels
        if label not in removed
    ] == list(macro.ordered_labels)

    expected_guarded = fixture["global_guarded_single_pauli_pool"]
    assert guarded.count == expected_guarded["count"]
    assert (
        guarded.ordered_labels_sha256
        == expected_guarded["ordered_labels_sha256"]
    )
    assert (
        guarded.ordered_pool_sha256
        == expected_guarded[
            "ordered_pool_sha256_post_intrinsic_identity_v3"
        ]
    )
    assert (
        guarded.source_parent_ordered_labels_sha256
        == expected_guarded["source_parent_ordered_labels_sha256"]
    )
