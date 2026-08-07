from __future__ import annotations

import hashlib
import json

from pipelines.reporting import (
    build_paper_i_ra_adapt_global_singleton_weak_weak_comparison_adapter
    as adapter_builder,
)


def _assert_self_digest(row: dict[str, object]) -> None:
    observed = row["sha256"]
    payload = dict(row)
    del payload["sha256"]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == observed


def test_emitted_weak_weak_adapter_has_closed_fixed_schema() -> None:
    adapter = json.loads(adapter_builder.DEFAULT_OUTPUT.read_text())

    assert adapter["schema"] == adapter_builder.ADAPTER_SCHEMA
    assert adapter["status"] == "passed"
    assert adapter["diagnostic_only"] is True
    assert adapter["paper_evidence_adopted"] is False
    assert adapter["regime_id"] == "weak_weak"
    assert adapter["nph"] == 3
    assert adapter["horizon"] == 50
    assert adapter["cross_arm_audit"]["status"] == "passed"
    assert adapter["cross_arm_audit"]["allowed_axis"] == "insertion_policy"
    assert adapter["cross_arm_audit"]["weak_weak_normalized_common_sha256"]

    assert [arm["insertion_policy"] for arm in adapter["arms"]] == [
        "append_commutation_reduced",
        "plateau_commutation",
    ]
    assert all(
        [point["k"] for point in arm["points"]] == list(range(51))
        for arm in adapter["arms"]
    )
    assert all(arm["qualification"]["status"] == "passed" for arm in adapter["arms"])
    assert all(arm["source"]["archive"]["sha256"] for arm in adapter["arms"])
    assert all(arm["source"]["result"]["sha256"] for arm in adapter["arms"])

    _assert_self_digest(adapter)
    for arm in adapter["arms"]:
        _assert_self_digest(arm)


def test_emitted_weak_weak_adapter_preserves_finished_run_metrics() -> None:
    adapter = json.loads(adapter_builder.DEFAULT_OUTPUT.read_text())
    append, plateau = adapter["arms"]

    assert append["effective_plateau"]["k"] == 50
    assert append["terminal"]["S_alg"] == 179_375
    assert append["terminal"]["N2q"] == 234
    assert append["terminal"]["D2q"] == 196
    assert append["terminal"]["Dc"] == 874
    assert append["insertion_counts"] == {
        "round_count": 50,
        "append_count": 50,
        "interior_count": 0,
        "first_interior_round": None,
    }

    assert plateau["effective_plateau"]["k"] == 49
    assert plateau["effective_plateau"]["S_alg"] == 848_329
    assert plateau["effective_plateau"]["N2q"] == 246
    assert plateau["effective_plateau"]["D2q"] == 208
    assert plateau["effective_plateau"]["Dc"] == 906
    assert plateau["terminal"]["S_alg"] == 903_285
    assert plateau["insertion_counts"] == {
        "round_count": 50,
        "append_count": 32,
        "interior_count": 18,
        "first_interior_round": 28,
    }

    comparison = adapter["comparison"]
    assert comparison["same_cutoff_exact_energy"] == append["qualification"][
        "exact_same_cutoff_energy"
    ]
    assert comparison["same_cutoff_exact_energy"] == plateau["qualification"][
        "exact_same_cutoff_energy"
    ]
    assert comparison["effective_plateau_round_delta"] == -1
    assert comparison["interior_insertion_count_delta"] == 18
