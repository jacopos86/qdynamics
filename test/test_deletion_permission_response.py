"""Round-replayed a_D attribution for deletion-bearing finalists."""

from __future__ import annotations

from pipelines.time_dynamics.diagnostics.deletion_permission_response import (
    attempt_rows,
    coefficient_map,
    summarize,
)


def _attempt(kind, removed, inserted, base, patched, reason="certified"):
    return {
        "kind": kind,
        "family": "singleton_d1" if removed else "singleton_d0",
        "removed_runtime_indices": list(removed),
        "inserted_selection": inserted,
        "base_runtime_parameter_count": base,
        "patched_runtime_parameter_count": patched,
        "reason": reason,
    }


def _doc():
    # Two coordinates enter t=0.04: A at angle 0.2 (c=0.5) and B at 0.001 (c=1).
    labels = ["fam::A::r0::xx", "fam::B::r1::yy"]
    theta_next = [0.2, 0.001]
    round0 = _attempt("insert", [], [["pauli:fam::C::r0::zz", 1]], 2, 3)  # C lands at index 1
    round1_reject = _attempt("delete", [0], [], 3, 2, reason="ray_distance_above_max")
    round1_commit = _attempt("exchange", [1], [["pauli:fam::D::r0::ee", 0]], 3, 3)  # deletes C (a_D = 0)
    points = [
        {
            "time": 0.0,
            "theta_runtime": [0.2, 0.001],
            "fixed_step": {"support_labels": labels, "theta_dot": [0.0, 0.0]},
            "integration_to_next": {"theta_next": theta_next},
            "patch_decision": {"metadata": {}},
        },
        {
            "time": 0.04,
            "theta_runtime": [0.0, 0.2, 0.001],
            "fixed_step": {"support_labels": ["D", "A", "B"]},
            "patch_decision": {
                "metadata": {
                    "selection_round_work": [
                        {"estimator_work": {"certification_attempts": [round0]}},
                        {"estimator_work": {"certification_attempts": [round1_reject, round1_commit]}},
                    ],
                    "attempts": [
                        {"kind": "delete", "removed_runtime_indices": [0], "ray_distance": 0.09, "smoothness_eta": 0.5},
                        {"kind": "exchange", "removed_runtime_indices": [1], "ray_distance": 0.0, "smoothness_eta": 0.0},
                    ],
                }
            },
        },
    ]
    return {
        "trajectory": {"points": points},
        "resume_state": {
            "coordinates": [
                {"label": "fam::A::r0::xx", "coeff_real": 0.5},
                {"label": "fam::B::r1::yy", "coeff_real": -1.0},
            ]
        },
    }


def test_a_d_is_attributed_on_the_round_entry_support_with_ray_values_joined():
    doc = _doc()
    rows, skipped = attempt_rows(doc, run="synthetic", coef=coefficient_map([doc]))
    assert dict(skipped) == {}
    assert [r["kind"] for r in rows] == ["delete", "exchange"]
    rejected, exchange = rows
    assert rejected["a_D"] == 0.2 * 0.5 and rejected["post_refit_ray"] == 0.09
    assert not rejected["certified"] and rejected["reason"] == "ray_distance_above_max"
    # index 1 at round 1 is the coordinate appended in round 0: zero angle.
    assert exchange["a_D"] == 0.0 and exchange["same_time_point_append"]
    assert exchange["certified"] and exchange["post_refit_ray"] == 0.0
    assert {r["entry_theta_source"] for r in rows} == {"recorded"}


def test_delta_theta_runs_fall_back_to_the_first_order_entry_estimate():
    doc = _doc()
    doc["trajectory"]["points"][0].pop("integration_to_next")
    doc["trajectory"]["points"][0]["fixed_step"]["theta_dot"] = [1.0, 0.0]
    rows, _ = attempt_rows(doc, run="synthetic", coef=coefficient_map([doc]))
    assert rows[0]["entry_theta_source"] == "euler_estimate"
    assert abs(rows[0]["a_D"] - (0.2 + 0.04 * 1.0) * 0.5) < 1e-12


def test_bins_count_zero_angle_deletions_separately():
    doc = _doc()
    rows, _ = attempt_rows(doc, run="synthetic", coef=coefficient_map([doc]))
    table = summarize(rows, ray_max=2.0e-3, velocity_max=1.0e-3)
    by_bin = {t["bin"]: t for t in table}
    assert by_bin["a_D = 0 (same-time-point append)"]["tested"] == 1
    assert by_bin["a_D >= 1e-1"]["tested"] == 1  # 0.2 * 0.5 = 0.1
    assert by_bin["a_D >= 1e-1"]["ray_within"] == 0
