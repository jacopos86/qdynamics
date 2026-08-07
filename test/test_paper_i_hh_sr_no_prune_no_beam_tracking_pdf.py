from __future__ import annotations

import hashlib
import json

import pytest

from pipelines.reporting import build_paper_i_hh_sr_no_prune_no_beam_tracking_pdf as tracker_builder
from pipelines.reporting import build_paper_i_hh_tracking_plateau_costs as plateau_builder
from pipelines.reporting.build_paper_i_hh_sr_no_prune_no_beam_tracking_pdf import (
    APPEND_PROJECTED_ROUTE_ID,
    PLATEAU_RULE_ID,
    TOP_SR_ROUTE_IDS,
    _attach_plateau_prefixes,
    _build_method_representation_comparison,
    _build_top_sr_append_plateau_comparison,
    _comparison_plateau_cost_table,
    _extract_trajectory,
    _plateau_cost_table,
    _tex,
)
from pipelines.reporting.build_paper_i_hh_tracking_plateau_costs import (
    RELATIVE_TOLERANCE,
    RULE_ID,
    SCHEMA,
    build_plateau_costs,
    select_plateau_prefix,
)


def _snake_payload(errors: list[float]) -> dict:
    return {
        "adapt_vqe": {
            "success": True,
            "exact_gs_energy": -1.0,
            "abs_delta_e": errors[-1],
            "history": [
                {
                    "outer_iteration": index,
                    "depth": 10 - index,
                    "delta_abs_current": error,
                    "energy_after_opt": -1.0 + error,
                }
                for index, error in enumerate(errors, start=1)
            ],
        },
        "settings": {"n_ph_max": 3},
    }


def _priority_sr_payload(*, trust_policy: str) -> dict:
    return {
        "settings": {
            "n_ph_max": 3,
            "sr_route_profile_contract_sha256": tracker_builder.PROJECTED_PHASE3_ROUTE_DIGEST,
            "historical_singleton_coordinate_solve_policy": (
                "supported_metric_projected_generalized_trust_v1"
            ),
            "historical_singleton_trust_region_update_policy": trust_policy,
            "phase3_response_coordinate_scope": "full_active_plus_singleton_v1",
            "sr_route_profile_contract": {
                "semantic_invariants": {
                    "phase3_support_projection_active": True,
                    "phase3_supported_whitening_active": False,
                    "phase3_supported_metric_inverse_sqrt_active": False,
                    "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
                    "accepted_refit_scope": "full_ansatz_v1",
                    "ordinary_phase2_novelty_multiplier_active": False,
                    "ordinary_phase3_novelty_multiplier_active": False,
                    "pruning_active": False,
                    "phase_live_hysteresis_enabled": False,
                }
            },
        },
        "adapt_vqe": {
            "accepted_refit": {
                "supported_fs_whitened": True,
                "full_ansatz": True,
            }
        },
    }


def test_priority_sr_page_requires_projected_nonwhitened_phase3() -> None:
    payload = _priority_sr_payload(
        trust_policy="displacement_calibrated_unbounded_v2"
    )
    tracker_builder._validate_priority_sr_payload(
        payload,
        regime="weak_weak",
        expected_route_digest=tracker_builder.PROJECTED_PHASE3_ROUTE_DIGEST,
        expected_trust_policy="displacement_calibrated_unbounded_v2",
    )

    payload["settings"]["sr_route_profile_contract"]["semantic_invariants"][
        "phase3_supported_whitening_active"
    ] = True
    with pytest.raises(RuntimeError, match="phase3_supported_whitening_active"):
        tracker_builder._validate_priority_sr_payload(
            payload,
            regime="weak_weak",
            expected_route_digest=tracker_builder.PROJECTED_PHASE3_ROUTE_DIGEST,
            expected_trust_policy="displacement_calibrated_unbounded_v2",
        )


def test_no_overlap_page_requires_zero_overlap_measurements_and_queries() -> None:
    receipt = {
        "status": "pass",
        "profile_contract_sha256": tracker_builder.NO_OVERLAP_TRUST_ROUTE_DIGEST,
        "target_controller_round": 50,
        "current_fake_marrakesh_metrics": {"N2q": 1, "D2q": 2, "Dc": 3},
        "scientific_evidence_validation": {
            "supported_rank_recorded_each_round": True,
            "active_prefix_estimator_ledger_receipts": {"closure_passed": True},
        },
        "projected_generalized_phase3_validation": {
            "status": "pass",
            "controller_rounds": 50,
            "supported_metric_whitening_active": False,
            "accepted_powell_refit_whitening_active": True,
        },
        "no_overlap_trust_validation": {
            "status": "pass",
            "controller_rounds": 50,
            "endpoint_overlap_measurement_count": 0,
            "endpoint_overlap_query_charge": 0,
            "accepted_powell_refit_whitening_active": True,
        },
    }
    assert tracker_builder._validate_priority_sr_receipt(
        receipt,
        regime="weak_weak",
        expected_route_digest=tracker_builder.NO_OVERLAP_TRUST_ROUTE_DIGEST,
        require_no_overlap=True,
    ) == {"N2q": 1, "D2q": 2, "Dc": 3}

    receipt["no_overlap_trust_validation"]["endpoint_overlap_measurement_count"] = 1
    with pytest.raises(RuntimeError, match="no-overlap receipt drift"):
        tracker_builder._validate_priority_sr_receipt(
            receipt,
            regime="weak_weak",
            expected_route_digest=tracker_builder.NO_OVERLAP_TRUST_ROUTE_DIGEST,
            require_no_overlap=True,
        )


def test_plateau_rule_selects_first_prefix_within_ten_percent_of_best() -> None:
    selected = select_plateau_prefix(
        _snake_payload([1.0, 0.30, 0.1088, 0.10, 0.099]),
        method="snake",
    )

    assert RULE_ID == PLATEAU_RULE_ID
    assert RELATIVE_TOLERANCE == pytest.approx(0.10)
    assert selected["k_pl"] == 3
    assert selected["error"] == pytest.approx(0.1088)
    assert selected["best_observed_error"] == pytest.approx(0.099)
    assert selected["threshold"] == pytest.approx(0.1089)


def test_plateau_builder_reuses_source_identical_cached_row(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(plateau_builder, "REPO_ROOT", tmp_path)
    source = {"path": "missing/source.json", "sha256": "a" * 64}
    tracker_path = tmp_path / "tracker.json"
    tracker_path.write_text(
        json.dumps(
            {
                "schema": "test_tracker",
                "routes": [
                    {
                        "id": "cached_route",
                        "results": {
                            "weak_weak": {
                                "status": "complete",
                                "trajectory": [{"round": 1, "error": 0.1}],
                                "source": source,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "plateau.json"
    cached_row = {
        "route_id": "cached_route",
        "regime": "weak_weak",
        "status": "complete",
        "rule": {"id": RULE_ID},
        "source": source,
        "k_pl": 1,
    }
    output_path.write_text(
        json.dumps({"schema": SCHEMA, "rows": [cached_row]}),
        encoding="utf-8",
    )

    payload = build_plateau_costs(
        tracker_json=tracker_path,
        output_json=output_path,
    )

    assert payload["rows"] == [cached_row]
    assert payload["summary"]["complete_prefix_count"] == 1


def test_comparator_plateau_never_calls_full_result_loader(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(plateau_builder, "REPO_ROOT", tmp_path)
    tracker_path = tmp_path / "tracker.json"
    source = {
        "path": "archive.tar.gz",
        "sha256": "a" * 64,
        "member": "job/result.json",
    }
    tracker_path.write_text(
        json.dumps(
            {
                "schema": "test_tracker",
                "routes": [
                    {
                        "id": "append_adapt_macro_nph3_7",
                        "results": {
                            "strong_strong_u8": {
                                "status": "complete",
                                "trajectory": [
                                    {"round": 1, "error": 0.2},
                                    {"round": 2, "error": 0.1},
                                ],
                                "source": source,
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        plateau_builder,
        "_read_source_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("comparator plateau must not fully load result.json")
        ),
    )
    monkeypatch.setattr(
        plateau_builder,
        "_comparator_plateau_prefix_streaming",
        lambda **_kwargs: (
            {
                "history_position": 2,
                "k_pl": 2,
                "outer_iteration": 2,
                "horizon": 2,
                "error": 0.1,
                "best_observed_error": 0.1,
                "threshold": 0.11,
                "active_depth": 2,
                "S_alg": 10,
                "qiskit": {"N2q": 1, "D2q": 1, "Dc": 2},
            },
            {
                "path": source["path"],
                "sha256": source["sha256"],
                "result_member": source["member"],
                "streaming_bounded_memory": True,
            },
        ),
    )

    payload = build_plateau_costs(
        tracker_json=tracker_path,
        output_json=tmp_path / "plateau.json",
    )

    assert payload["rows"][0]["route_id"] == "append_adapt_macro_nph3_7"
    assert payload["rows"][0]["source"]["streaming_bounded_memory"] is True


def test_snake_trajectory_uses_controller_round_not_pruned_active_depth() -> None:
    trajectory = _extract_trajectory(_snake_payload([0.2, 0.1]))

    assert [point["round"] for point in trajectory["trajectory"]] == [1, 2]
    assert [point["active_depth"] for point in trajectory["trajectory"]] == [9, 8]


def test_plateau_attachment_fails_closed_on_source_hash_drift(tmp_path) -> None:
    source = tmp_path / "result.json"
    source.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    route = {
        "id": "route",
        "label": "route",
        "results": {
            regime: {
                "trajectory": ([{"round": 1, "error": 0.1}] if regime == "weak_weak" else []),
                "source": (
                    {"path": str(source), "sha256": digest}
                    if regime == "weak_weak"
                    else None
                ),
                "status": "complete" if regime == "weak_weak" else "missing",
            }
            for regime in (
                "weak_weak",
                "intermediate_weak",
                "strong_weak_u8",
                "weak_strong",
                "intermediate_strong",
                "strong_strong_u8",
            )
        },
    }
    plateau = tmp_path / "plateau.json"
    plateau.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "rule": {"id": RULE_ID},
                "compile_policy": {},
                "summary": {},
                "rows": [
                    {
                        "route_id": "route",
                        "regime": "weak_weak",
                        "status": "complete",
                        "rule": {"id": RULE_ID},
                        "k_pl": 1,
                        "error": 0.1,
                        "S_alg": 7,
                        "qiskit": {"N2q": 1, "D2q": 1, "Dc": 2},
                        "source": {"path": str(source), "sha256": "0" * 64},
                    }
                ],
                "unresolved": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not match displayed result"):
        _attach_plateau_prefixes([route], plateau_json=plateau, sources=[])


def test_cost_arm_plateau_attachment_preserves_exact_prefix_unavailable(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(tracker_builder, "REPO_ROOT", tmp_path)
    regimes = (
        "weak_weak",
        "intermediate_weak",
        "strong_weak_u8",
        "weak_strong",
        "intermediate_strong",
        "strong_strong_u8",
    )
    route = {
        "id": "sr_macro_beam3x2_fs_prune_symmetric_cost_nph3_7",
        "results": {
            regime: {
                "trajectory": (
                    [{"round": 1, "error": 0.1}] if regime == "weak_weak" else []
                ),
                "status": "complete" if regime == "weak_weak" else "missing",
            }
            for regime in regimes
        },
    }
    plateau = tmp_path / "plateau.json"
    plateau.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "rule": {"id": RULE_ID},
                "rows": [],
                "unresolved": [
                    {
                        "route_id": route["id"],
                        "regime": "weak_weak",
                        "status": "exact_prefix_unavailable",
                        "reason": "selected prefix predates executable terminal checkpoint",
                        "k_pl": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _attach_plateau_prefixes([route], plateau_json=plateau, sources=[])

    assert route["plateau"]["weak_weak"]["status"] == "exact_prefix_unavailable"


def test_plateau_table_keeps_prefix_fields_aligned() -> None:
    route = {
        "results": {
            regime: {"n_ph": 3 if index < 3 else 7}
            for index, regime in enumerate(
                (
                    "weak_weak",
                    "intermediate_weak",
                    "strong_weak_u8",
                    "weak_strong",
                    "intermediate_strong",
                    "strong_strong_u8",
                )
            )
        },
        "plateau": {
            "weak_weak": {
                "k_pl": 9,
                "error": 1.2e-7,
                "S_alg": 123,
                "qiskit": {"N2q": 11, "D2q": 7, "Dc": 19},
            }
        },
    }

    table = _plateau_cost_table(route)

    assert r"$k_{\rm pl}$" in table
    assert "weak--weak & 3 & 9 & 1.200e-07 & 123 & 11 & 7 & 19" in table


def _comparison_route(
    route_id: str,
    *,
    multiplier: float,
    complete_regimes: tuple[str, ...],
) -> dict:
    regimes = (
        "weak_weak",
        "intermediate_weak",
        "strong_weak_u8",
        "weak_strong",
        "intermediate_strong",
        "strong_strong_u8",
    )
    return {
        "id": route_id,
        "label": route_id,
        "results": {
            regime: {
                "n_ph": 3 if index < 3 else 7,
                "rounds": 50,
                "s_alg": 1000 + index,
                "terminal_error": multiplier * (index + 1),
                "trajectory": (
                    [
                        {"round": 1, "error": 10 * multiplier * (index + 1)},
                        {"round": 50, "error": multiplier * (index + 1)},
                    ]
                    if regime in complete_regimes
                    else []
                ),
            }
            for index, regime in enumerate(regimes)
        },
        "plateau": {
            regime: (
                {
                    "status": "complete",
                    "k_pl": index + 1,
                    "error": multiplier * (index + 1),
                    "S_alg": 100 + index,
                    "qiskit": {
                        "N2q": 10 + index,
                        "D2q": 20 + index,
                        "Dc": 30 + index,
                    },
                }
                if regime in complete_regimes
                else {"status": "unresolved"}
            )
            for index, regime in enumerate(regimes)
        },
        "target_energy": {
            regime: (
                {
                    "status": "complete",
                    "k_target": index + 1,
                    "error": multiplier * (index + 1),
                    "S_alg": 100 + index,
                    "qiskit": {
                        "N2q": 10 + index,
                        "D2q": 20 + index,
                        "Dc": 30 + index,
                    },
                }
                if regime in complete_regimes and index % 2 == 0
                else {"status": "threshold_not_reached"}
            )
            for index, regime in enumerate(regimes)
        },
        "costs": {
            regime: {
                "N2q": 110 + index,
                "D2q": 120 + index,
                "Dc": 130 + index,
            }
            for index, regime in enumerate(regimes)
        },
    }


def test_top_sr_append_comparison_is_pinned_to_corrected_matched_routes() -> None:
    all_regimes = (
        "weak_weak",
        "intermediate_weak",
        "strong_weak_u8",
        "weak_strong",
        "intermediate_strong",
        "strong_strong_u8",
    )
    routes = [
        _comparison_route(TOP_SR_ROUTE_IDS[0], multiplier=1.0e-7, complete_regimes=all_regimes),
        _comparison_route(TOP_SR_ROUTE_IDS[1], multiplier=2.0e-7, complete_regimes=all_regimes),
        _comparison_route(
            APPEND_PROJECTED_ROUTE_ID,
            multiplier=3.0e-7,
            complete_regimes=all_regimes[:3],
        ),
        _comparison_route("historical_pre_correction", multiplier=1.0e-12, complete_regimes=all_regimes),
    ]

    comparison = _build_top_sr_append_plateau_comparison(routes)

    assert comparison["route_ids"] == [*TOP_SR_ROUTE_IDS, APPEND_PROJECTED_ROUTE_ID]
    assert comparison["common_regimes"] == list(all_regimes[:3])
    assert comparison["unresolved_append_regimes"] == list(all_regimes[3:])
    assert "historical_pre_correction" not in comparison["route_ids"]
    assert comparison["marker_policy"] == "one method-specific marker per curve at exact k_pl"


def test_comparison_table_reads_exact_plateau_costs_not_endpoint_costs() -> None:
    route = _comparison_route(
        TOP_SR_ROUTE_IDS[0],
        multiplier=1.0e-7,
        complete_regimes=("weak_weak",),
    )
    route["costs"] = {"weak_weak": {"N2q": 999, "D2q": 999, "Dc": 999}}

    table = _comparison_plateau_cost_table(route, ("weak_weak",))

    assert "WW & 1 & 1.000e-07 & 100 & 10 & 20 & 30" in table
    assert "999" not in table


@pytest.mark.parametrize(
    ("representation", "route_ids"),
    tuple(tracker_builder.METHOD_REPRESENTATION_ROUTE_IDS.items()),
)
def test_three_method_representation_comparison_uses_target_or_terminal(
    representation: str,
    route_ids: tuple[str, ...],
) -> None:
    all_regimes = tuple(tracker_builder.REGIMES)
    routes = [
        _comparison_route(
            route_id,
            multiplier=(index + 1) * 1.0e-5,
            complete_regimes=all_regimes,
        )
        for index, route_id in enumerate(route_ids)
    ]

    comparison = _build_method_representation_comparison(
        routes,
        representation=representation,
    )

    assert comparison["route_ids"] == list(route_ids)
    assert comparison["hit_counts"] == {route_id: 3 for route_id in route_ids}
    for route_id in route_ids:
        assert comparison["rows"][route_id]["weak_weak"]["endpoint"] == "target_crossing"
        assert comparison["rows"][route_id]["intermediate_weak"]["endpoint"] == "terminal_nonhit"
        assert comparison["rows"][route_id]["intermediate_weak"]["round"] == 50
        assert comparison["rows"][route_id]["intermediate_weak"]["qiskit"]["N2q"] == 111


def test_tex_escaping_does_not_reescape_generated_sequences() -> None:
    assert _tex("a_b&c") == r"a\_b\&c"
    assert _tex("{") == r"\{"


def test_late_comparator_archive_overrides_are_explicit() -> None:
    root = tracker_builder.COMPARATOR_LATE_FETCH

    assert tracker_builder.APPEND_MACRO_ARCHIVES["strong_strong_u8"] == root / (
        "append_macro__strong_strong__r50_transfer.tar.gz"
    )
    assert tracker_builder.GEO_PROJECTED_ARCHIVES["weak_strong"] == root / (
        "geo_projected_singleton__weak_strong__r50_transfer.tar.gz"
    )
    successor = tracker_builder.GEO_PROJECTED_SUCCESSOR_FETCH
    assert tracker_builder.GEO_PROJECTED_ARCHIVES["intermediate_strong"] == successor / (
        "geo_projected_singleton__intermediate_strong__r50_transfer.tar.gz"
    )
    assert tracker_builder.GEO_PROJECTED_ARCHIVES["strong_strong_u8"] == successor / (
        "geo_projected_singleton__strong_strong__r50_transfer.tar.gz"
    )
    assert tracker_builder.APPEND_PROJECTED_ARCHIVES["weak_strong"] == root / (
        "append_projected_singleton__weak_strong__r50_transfer.tar.gz"
    )
    assert tracker_builder.APPEND_PROJECTED_ARCHIVES["strong_strong_u8"] == root / (
        "append_projected_singleton__strong_strong__r50_transfer.tar.gz"
    )


def test_source_value_anchor_is_manifest_only_and_exactly_source_locked(
    tmp_path, monkeypatch
) -> None:
    archive = tmp_path / "8900512.0__weak_weak_transfer.tar.gz"
    receipt_path = tmp_path / "8900512.0__weak_weak_validation_receipt.json"
    audit_path = tmp_path / "source_locked_sensitivity_audit.json"
    archive.write_bytes(b"source-value-anchor")
    result_sha = "1" * 64
    receipt = {
        "status": "pass",
        "profile_contract_sha256": tracker_builder.CORRECTED_MAIN_ROUTE_DIGEST,
        "result_sha256": result_sha,
        "target_controller_round": 50,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    audit = {
        "schema": "source_locked_sensitivity_audit_v1",
        "status": "anchor_pass_fanout_authorized",
        "fanout_authorized": True,
        "anchor": {
            "anchor_reproduces_source": True,
            "anchor_transfer_archive": archive.name,
            "anchor_transfer_archive_sha256": archive_sha,
            "anchor_validation_receipt": receipt_path.name,
            "anchor_validation_receipt_sha256": receipt_sha,
            "anchor_result_sha256": result_sha,
            "controller_energy_history_exact_match": True,
            "metric_abs_diff": 0.0,
            "non_swept_settings_diff": [],
            "operator_sequence_match": True,
            "settings_exact_match": True,
            "terminal_abs_delta_e": 2.99e-7,
            "value": "supported_metric_whitened_eigh_v1",
        },
        "source": {
            "route_contract_sha256": tracker_builder.CORRECTED_MAIN_ROUTE_DIGEST,
        },
        "sweep": {"variable": "historical_singleton_coordinate_solve_policy"},
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.setattr(tracker_builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tracker_builder, "PROJECTED_PARENT_ANCHOR_ARCHIVE", archive)
    monkeypatch.setattr(
        tracker_builder,
        "PROJECTED_PARENT_ANCHOR_VALIDATION",
        receipt_path,
    )
    monkeypatch.setattr(tracker_builder, "PROJECTED_PARENT_ANCHOR_AUDIT", audit_path)
    monkeypatch.setattr(
        tracker_builder,
        "PROJECTED_PARENT_ANCHOR_AUDIT_SHA256",
        hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    )
    tracker_builder._sha256.cache_clear()
    sources = []

    notes = tracker_builder._source_lock_notes(sources)

    assert notes[0]["status"] == "pass"
    assert notes[0]["duplicate_of_route_id"] == (
        "corrected_main_hysteresis_disabled_nph3_7"
    )
    assert notes[0]["display_policy"] == (
        "manifest_only_no_duplicate_trajectory_or_page"
    )
    assert {source["path"] for source in sources} == {
        archive.name,
        receipt_path.name,
        audit_path.name,
    }


def test_inventory_extension_reads_only_four_compact_late_rows(
    tmp_path, monkeypatch
) -> None:
    def route(route_id: str) -> dict:
        return {
            "id": route_id,
            "marker": f"preserve-{route_id}",
            "results": {
                regime: {"status": "base", "slot": f"result-{regime}"}
                for regime in tracker_builder.REGIMES
            },
            "costs": {
                regime: {"slot": f"cost-{regime}"}
                for regime in tracker_builder.REGIMES
            },
        }

    base_routes = [
        {"id": "keep", "marker": "unchanged"},
        route("append_adapt_macro_nph3_7"),
        route("geo_adapt_projected_singleton_nph3_7"),
        route("append_adapt_projected_singleton_nph3_7"),
        {"id": "sr_guarded_singleton_no_lanes_nph3_7", "marker": "unchanged"},
    ]
    base_path = tmp_path / "base.json"
    base_path.write_text(
        json.dumps(
            {
                "schema": tracker_builder.SCHEMA,
                "routes": base_routes,
                "sources": [{"path": "preserved", "sha256": "a" * 64}],
                "source_lock_notes": [{"status": "pass", "marker": "preserved"}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        tracker_builder,
        "_source_record",
        lambda path: {"path": str(path), "sha256": "b" * 64},
    )

    monkeypatch.setattr(
        tracker_builder,
        "_build_refreshable_comparator_routes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("base refresh must not rebuild old comparator archives")
        ),
    )
    monkeypatch.setattr(
        tracker_builder,
        "_build_pool_complement_routes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("base refresh must preserve existing pool routes")
        ),
    )
    monkeypatch.setattr(
        tracker_builder,
        "_tar_json_members",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("base refresh must not materialize old result.json members")
        ),
    )
    monkeypatch.setattr(
        tracker_builder,
        "_build_pass_only_cost_arm_routes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        tracker_builder,
        "_build_priority_sr_routes",
        lambda *_args, **_kwargs: [],
    )
    calls = []

    def compact_summary(**kwargs):
        calls.append((kwargs["expected_variant"], kwargs["regime"]))
        token = f"{kwargs['expected_variant']}-{kwargs['regime']}"
        return (
            {"status": "complete", "trajectory": [{"round": 1, "error": 0.1}], "token": token},
            {"N2q": 1, "D2q": 2, "Dc": 3, "token": token},
            [{"path": f"late-{token}", "sha256": "d" * 64}],
        )

    monkeypatch.setattr(
        tracker_builder,
        "_comparator_tracking_summary",
        compact_summary,
    )

    output = tracker_builder.extend_inventory_from_existing_report(
        tmp_path / "out",
        base_report_json=base_path,
    )
    inventory = json.loads(output["json"].read_text(encoding="utf-8"))

    assert [route["id"] for route in inventory["routes"]] == [
        "keep",
        "append_adapt_macro_nph3_7",
        "geo_adapt_projected_singleton_nph3_7",
        "append_adapt_projected_singleton_nph3_7",
        "sr_guarded_singleton_no_lanes_nph3_7",
    ]
    assert inventory["routes"][0] == base_routes[0]
    assert inventory["routes"][-1] == base_routes[-1]
    refreshed = {route["id"]: route for route in inventory["routes"]}
    expected_late = {
        ("append_adapt_macro_nph3_7", "strong_strong_u8"),
        ("geo_adapt_projected_singleton_nph3_7", "weak_strong"),
        ("geo_adapt_projected_singleton_nph3_7", "intermediate_strong"),
        ("geo_adapt_projected_singleton_nph3_7", "strong_strong_u8"),
        ("append_adapt_projected_singleton_nph3_7", "weak_strong"),
        ("append_adapt_projected_singleton_nph3_7", "intermediate_strong"),
        ("append_adapt_projected_singleton_nph3_7", "strong_strong_u8"),
    }
    assert set(calls) == {
        ("append_macro", "strong_strong_u8"),
        ("geo_projected_singleton", "weak_strong"),
        ("geo_projected_singleton", "intermediate_strong"),
        ("geo_projected_singleton", "strong_strong_u8"),
        ("append_projected_singleton", "weak_strong"),
        ("append_projected_singleton", "intermediate_strong"),
        ("append_projected_singleton", "strong_strong_u8"),
    }
    base_map = {route["id"]: route for route in base_routes if "results" in route}
    for route_id, base_route in base_map.items():
        assert refreshed[route_id]["marker"] == base_route["marker"]
        for regime in tracker_builder.REGIMES:
            if (route_id, regime) in expected_late:
                assert refreshed[route_id]["results"][regime]["status"] == "complete"
                assert refreshed[route_id]["costs"][regime]["N2q"] == 1
            else:
                assert refreshed[route_id]["results"][regime] == base_route["results"][regime]
                assert refreshed[route_id]["costs"][regime] == base_route["costs"][regime]
    assert inventory["sources"][0] == {"path": "preserved", "sha256": "a" * 64}
    assert {source["path"] for source in inventory["sources"][-7:]} == {
        "late-append_macro-strong_strong_u8",
        "late-geo_projected_singleton-weak_strong",
        "late-geo_projected_singleton-intermediate_strong",
        "late-geo_projected_singleton-strong_strong_u8",
        "late-append_projected_singleton-weak_strong",
        "late-append_projected_singleton-intermediate_strong",
        "late-append_projected_singleton-strong_strong_u8",
    }
    assert inventory["source_lock_notes"][0]["status"] == "pass"
    assert {
        note["route_id"] for note in inventory["pending_validation_notes"]
    } == {
        "sr_macro_beam3x2_fs_prune_symmetric_cost_nph3_7",
        "sr_macro_beam3x2_fs_prune_one_sided_cost_nph3_7",
    }
    assert not any(
        route["id"].startswith("sr_macro_beam3x2_fs_prune_")
        for route in inventory["routes"]
    )


def test_late_comparator_refresh_fails_closed_on_missing_base_route() -> None:
    with pytest.raises(RuntimeError, match="lacks late comparator route"):
        tracker_builder._refresh_late_comparator_rows_from_base([], sources=[])
