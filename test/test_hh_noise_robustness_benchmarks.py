from __future__ import annotations

from pipelines.exact_bench.hh_noise_robustness_seq_report import (
    _build_mitigation_config,
    _build_summary,
    _build_symmetry_mitigation_config,
    _collect_noisy_benchmark_rows,
    _compute_time_dynamics_proxy_cost,
    _disabled_hardcoded_superset_meta,
    _noise_config_caption,
    _noise_style_legend_lines,
    _parse_noisy_methods_csv,
    _validate_pool_b_strict_composition,
    parse_args,
)


def test_parse_args_noisy_benchmark_flags() -> None:
    args = parse_args(
        [
            "--noisy-methods",
            "suzuki2,cfqm4",
            "--benchmark-active-coeff-tol",
            "1e-9",
        ]
    )
    assert str(args.noisy_methods) == "suzuki2,cfqm4"
    assert float(args.benchmark_active_coeff_tol) == 1e-9
    assert bool(args.disable_time_dynamics) is False


def test_parse_args_mitigation_defaults_and_values() -> None:
    args = parse_args([])
    assert str(args.mitigation) == "none"
    assert str(args.symmetry_mitigation_mode) == "off"
    assert args.zne_scales is None
    assert args.dd_sequence is None

    args = parse_args(
        [
            "--mitigation",
            "dd",
            "--symmetry-mitigation-mode",
            "projector_renorm_v1",
            "--zne-scales",
            "1.0,2.0",
            "--dd-sequence",
            "XY4",
        ]
    )
    assert str(args.mitigation) == "dd"
    assert str(args.symmetry_mitigation_mode) == "projector_renorm_v1"
    assert str(args.zne_scales) == "1.0,2.0"
    assert str(args.dd_sequence) == "XY4"


def test_parse_args_disable_time_dynamics_flag() -> None:
    args = parse_args(["--disable-time-dynamics"])
    assert bool(args.disable_time_dynamics) is True


def test_mitigation_schema_defaults_and_caption() -> None:
    mit = _build_mitigation_config(mitigation="none", zne_scales=None, dd_sequence=None)
    sym = _build_symmetry_mitigation_config(mode="postselect_diag_v1", L=2, ordering="blocked")
    assert mit == {"mode": "none", "zne_scales": [], "dd_sequence": None}
    assert sym == {
        "mode": "postselect_diag_v1",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }

    caption = _noise_config_caption(
        {
            "shots": 2048,
            "oracle_repeats": 4,
            "oracle_aggregate": "mean",
            "mitigation_config": mit,
            "symmetry_mitigation_config": sym,
        },
        "shots",
    )
    assert "mitigation=none" in caption
    assert "symmetry=postselect_diag_v1" in caption


def test_parse_noisy_methods_csv_validation() -> None:
    assert _parse_noisy_methods_csv("suzuki2,cfqm4,suzuki2") == ["suzuki2", "cfqm4"]


def test_pool_b_enforcement_passes_for_exact_family_set() -> None:
    audit = _validate_pool_b_strict_composition(
        {
            "raw_sizes": {"uccsd": 4, "hva": 6, "paop_full": 3},
            "dedup_source_presence_counts": {"uccsd": 4, "hva": 5, "paop_full": 3},
        }
    )
    assert bool(audit["passed"]) is True
    assert list(audit["required_families"]) == ["uccsd_lifted", "hva", "paop_full"]


def test_pool_b_enforcement_fails_on_missing_family() -> None:
    try:
        _validate_pool_b_strict_composition(
            {
                "raw_sizes": {"uccsd": 4, "hva": 6},
                "dedup_source_presence_counts": {"uccsd": 4, "hva": 5},
            }
        )
        raise AssertionError("Expected ValueError for missing Pool B family.")
    except ValueError as exc:
        assert "Pool B composition mismatch" in str(exc)


def test_cfqm_proxy_cost_is_deterministic() -> None:
    kwargs = dict(
        method="cfqm4",
        t_final=1.0,
        trotter_steps=4,
        drive_t0=0.0,
        drive_time_sampling="midpoint",
        ordered_labels_exyz=["ee", "xz", "yy"],
        static_coeff_map_exyz={"ee": 1.0 + 0.0j, "xz": 0.2 + 0.0j, "yy": -0.1 + 0.0j},
        drive_provider_exyz=None,
        active_coeff_tol=1e-12,
        coeff_drop_abs_tol=0.0,
    )
    c1 = _compute_time_dynamics_proxy_cost(**kwargs)
    c2 = _compute_time_dynamics_proxy_cost(**kwargs)
    assert c1 == c2
    assert int(c1["cx_proxy_total"]) >= 0
    assert int(c1["term_exp_count_total"]) > 0


def test_suzuki_and_cfqm_proxy_cost_sanity() -> None:
    base_kwargs = dict(
        t_final=1.0,
        trotter_steps=4,
        drive_t0=0.0,
        drive_time_sampling="midpoint",
        ordered_labels_exyz=["ee", "xz", "yy"],
        static_coeff_map_exyz={"ee": 1.0 + 0.0j, "xz": 0.2 + 0.0j, "yy": -0.1 + 0.0j},
        drive_provider_exyz=None,
        active_coeff_tol=1e-12,
        coeff_drop_abs_tol=0.0,
    )
    suz = _compute_time_dynamics_proxy_cost(method="suzuki2", **base_kwargs)
    cfq = _compute_time_dynamics_proxy_cost(method="cfqm4", **base_kwargs)
    for rec in (suz, cfq):
        assert set(rec.keys()) == {
            "term_exp_count_total",
            "pauli_rot_count_total",
            "cx_proxy_total",
            "sq_proxy_total",
            "depth_proxy_total",
        }
        assert int(rec["term_exp_count_total"]) >= 0
        assert int(rec["depth_proxy_total"]) == int(rec["pauli_rot_count_total"])


def test_collect_noisy_benchmark_rows_schema_and_values() -> None:
    dyn_noisy = {
        "profiles": {
            "static": {
                "methods": {
                    "suzuki2": {
                        "modes": {
                            "shots": {
                                "success": True,
                                "delta_uncertainty": {
                                    "energy_total": {
                                        "max_abs_delta": 0.02,
                                        "max_abs_delta_over_stderr": 4.0,
                                        "mean_abs_delta_over_stderr": 2.5,
                                    }
                                },
                                "benchmark_cost": {
                                    "term_exp_count_total": 100,
                                    "pauli_rot_count_total": 100,
                                    "cx_proxy_total": 220,
                                    "sq_proxy_total": 340,
                                    "depth_proxy_total": 100,
                                },
                                "benchmark_runtime": {
                                    "wall_total_s": 1.25,
                                    "oracle_eval_s_total": 0.55,
                                    "oracle_calls_total": 120,
                                },
                            }
                        }
                    },
                    "cfqm4": {
                        "modes": {
                            "shots": {
                                "success": True,
                                "delta_uncertainty": {
                                    "energy_total": {
                                        "max_abs_delta": 0.015,
                                        "max_abs_delta_over_stderr": 3.0,
                                        "mean_abs_delta_over_stderr": 2.0,
                                    }
                                },
                                "benchmark_cost": {
                                    "term_exp_count_total": 88,
                                    "pauli_rot_count_total": 88,
                                    "cx_proxy_total": 180,
                                    "sq_proxy_total": 300,
                                    "depth_proxy_total": 88,
                                },
                                "benchmark_runtime": {
                                    "wall_total_s": 1.10,
                                    "oracle_eval_s_total": 0.48,
                                    "oracle_calls_total": 120,
                                },
                            }
                        }
                    },
                },
                "modes": {},
            }
        }
    }
    rows = _collect_noisy_benchmark_rows(dyn_noisy)
    assert len(rows) == 2
    methods = {str(r["method"]) for r in rows}
    assert methods == {"suzuki2", "cfqm4"}
    for row in rows:
        assert set(row.keys()) == {
            "profile",
            "method",
            "mode",
            "term_exp_count_total",
            "pauli_rot_count_total",
            "cx_proxy_total",
            "sq_proxy_total",
            "depth_proxy_total",
            "wall_total_s",
            "oracle_eval_s_total",
            "oracle_calls_total",
            "max_abs_delta",
            "max_abs_delta_over_stderr",
            "mean_abs_delta_over_stderr",
        }


def test_disabled_hardcoded_superset_metadata_and_summary_shape() -> None:
    hardcoded = _disabled_hardcoded_superset_meta()
    assert bool(hardcoded["disabled"]) is True
    assert hardcoded["profiles"] == {}
    assert "final-only dynamics" in str(hardcoded.get("reason", ""))

    payload = {
        "stage_pipeline": {
            "warm_start": {"delta_abs": 0.1, "stop_reason": "warm_done"},
            "adapt_pool_b": {"delta_abs": 0.01, "stop_reason": "adapt_done"},
            "conventional_vqe": {"delta_abs": 0.001, "stop_reason": "final_done"},
        },
        "hardcoded_superset": hardcoded,
        "dynamics_noisy": {
            "profiles": {
                "static": {
                    "modes": {},
                    "methods": {
                        "suzuki2": {
                            "modes": {
                                "shots": {
                                    "success": True,
                                    "delta_uncertainty": {
                                        "energy_total": {
                                            "max_abs_delta": 0.01,
                                            "max_abs_delta_over_stderr": 5.0,
                                            "mean_abs_delta_over_stderr": 3.5,
                                        }
                                    },
                                }
                            }
                        }
                    },
                }
            }
        },
        "dynamics_benchmarks": {"rows": [{"profile": "static", "method": "suzuki2", "mode": "shots"}]},
    }
    summary = _build_summary(payload)
    assert int(summary["noisy_method_modes_total"]) == 1
    assert int(summary["noisy_method_modes_completed"]) == 1
    assert int(summary["dynamics_benchmark_rows"]) == 1
    assert float(summary["max_abs_delta"]) == 0.01
    assert float(summary["max_abs_delta_over_stderr"]) == 5.0
    assert float(summary["mean_abs_delta_over_stderr"]) == 3.5


def test_summary_uses_noisy_final_audit_when_dynamics_disabled() -> None:
    payload = {
        "stage_pipeline": {
            "warm_start": {"delta_abs": 0.1, "stop_reason": "warm_done"},
            "adapt_pool_b": {"delta_abs": 0.01, "stop_reason": "adapt_done"},
            "conventional_vqe": {"delta_abs": 0.001, "stop_reason": "final_done"},
        },
        "dynamics_noisy": {"profiles": {}},
        "noisy_final_audit": {
            "profiles": {
                "static": {
                    "modes": {
                        "shots": {
                            "success": True,
                            "delta_uncertainty": {
                                "energy_total": {
                                    "max_abs_delta": 0.02,
                                    "max_abs_delta_over_stderr": 4.5,
                                    "mean_abs_delta_over_stderr": 2.8,
                                }
                            },
                        },
                        "aer_noise": {
                            "success": False,
                            "reason": "env_blocked",
                        },
                    }
                }
            }
        },
        "dynamics_benchmarks": {"rows": []},
    }
    summary = _build_summary(payload)
    assert int(summary["noisy_audit_modes_total"]) == 2
    assert int(summary["noisy_audit_modes_completed"]) == 1
    assert float(summary["noisy_audit_max_abs_delta"]) == 0.02
    assert float(summary["noisy_audit_max_abs_delta_over_stderr"]) == 4.5
    assert float(summary["noisy_audit_mean_abs_delta_over_stderr"]) == 2.8
    # Combined summary fields should still be populated even with no trajectories.
    assert float(summary["max_abs_delta"]) == 0.02


def test_noise_style_legend_semantics_tokens_present() -> None:
    text = "\n".join(_noise_style_legend_lines())
    assert "Δ(noisy-ideal)" in text
    assert "noiseless (final-seed Suzuki-2)" in text
