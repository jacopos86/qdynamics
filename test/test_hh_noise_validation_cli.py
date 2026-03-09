from __future__ import annotations

import pytest

from pipelines.exact_bench.hh_noise_hardware_validation import (
    _build_mitigation_config_from_args,
    _build_symmetry_mitigation_config_from_args,
    _apply_defaults_and_minimums,
    parse_args,
)


def test_hh_defaults_applied_from_minimum_table_l2_nph1() -> None:
    args = parse_args(["--L", "2"])
    args = _apply_defaults_and_minimums(args)

    assert int(args.vqe_reps) == 2
    assert int(args.vqe_restarts) == 3
    assert int(args.vqe_maxiter) == 800
    assert int(args.trotter_steps) == 64


def test_hh_under_minimum_rejected_without_smoke_flag() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--vqe-reps",
            "1",
            "--vqe-restarts",
            "1",
            "--vqe-maxiter",
            "100",
            "--trotter-steps",
            "8",
        ]
    )
    with pytest.raises(ValueError):
        _apply_defaults_and_minimums(args)


def test_hh_under_minimum_allowed_with_smoke_flag() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--vqe-reps",
            "1",
            "--vqe-restarts",
            "1",
            "--vqe-maxiter",
            "100",
            "--trotter-steps",
            "8",
            "--smoke-test-intentionally-weak",
        ]
    )
    args = _apply_defaults_and_minimums(args)

    assert int(args.vqe_reps) == 1
    assert int(args.vqe_restarts) == 1
    assert int(args.vqe_maxiter) == 100
    assert int(args.trotter_steps) == 8


def test_hubbard_under_minimum_rejected_without_smoke_flag() -> None:
    args = parse_args(
        [
            "--problem",
            "hubbard",
            "--ansatz",
            "hva",
            "--L",
            "4",
            "--vqe-reps",
            "1",
            "--vqe-restarts",
            "1",
            "--vqe-maxiter",
            "100",
            "--trotter-steps",
            "16",
        ]
    )
    with pytest.raises(ValueError):
        _apply_defaults_and_minimums(args)


def test_cli_parses_fallback_flags() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--no-allow-aer-fallback",
            "--no-omp-shm-workaround",
        ]
    )
    assert bool(args.allow_aer_fallback) is False
    assert bool(args.omp_shm_workaround) is False


def test_cli_parses_legacy_parity_flags() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--legacy-reference-json",
            "artifacts/json/hc_hh_L2_static_t1.0_U2.0_g1.0_nph1.json",
            "--legacy-parity-tol",
            "1e-10",
            "--output-compare-plot",
            "artifacts/pdf/hh_noise_cmp.png",
            "--compare-observables",
            "energy_static_trotter,doublon_trotter",
        ]
    )
    assert str(args.legacy_reference_json).endswith(
        "artifacts/json/hc_hh_L2_static_t1.0_U2.0_g1.0_nph1.json"
    )
    assert float(args.legacy_parity_tol) == pytest.approx(1e-10)
    assert str(args.output_compare_plot).endswith("artifacts/pdf/hh_noise_cmp.png")
    assert str(args.compare_observables) == "energy_static_trotter,doublon_trotter"


def test_cli_parses_mitigation_flags_defaults_and_values() -> None:
    args = parse_args(["--L", "2"])
    assert str(args.mitigation) == "none"
    assert args.zne_scales is None
    assert args.dd_sequence is None
    assert _build_mitigation_config_from_args(args) == {
        "mode": "none",
        "zne_scales": [],
        "dd_sequence": None,
    }

    args = parse_args(
        [
            "--L",
            "2",
            "--mitigation",
            "zne",
            "--zne-scales",
            "1.0,2.0,3.0",
            "--dd-sequence",
            "XY4",
        ]
    )
    assert str(args.mitigation) == "zne"
    assert str(args.zne_scales) == "1.0,2.0,3.0"
    assert str(args.dd_sequence) == "XY4"


def test_cli_parses_symmetry_mitigation_flags_defaults_and_values() -> None:
    args = parse_args(["--L", "2"])
    assert str(args.symmetry_mitigation_mode) == "off"
    assert _build_symmetry_mitigation_config_from_args(args) == {
        "mode": "off",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }

    args = parse_args(
        [
            "--L",
            "2",
            "--symmetry-mitigation-mode",
            "postselect_diag_v1",
        ]
    )
    assert str(args.symmetry_mitigation_mode) == "postselect_diag_v1"
    assert _build_symmetry_mitigation_config_from_args(args) == {
        "mode": "postselect_diag_v1",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }
