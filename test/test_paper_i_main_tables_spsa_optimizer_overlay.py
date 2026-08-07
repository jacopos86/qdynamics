from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.exact_bench.generic_static_benchmark import run_single
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (
    PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS,
    PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_HEA_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES,
    PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    PAPER_I_MAIN_TABLES_SPSA_SCHEDULE_TSV_FIELDS,
)


def test_paper_i_main_tables_spsa_schedule_env_constants_are_additive() -> None:
    assert PAPER_I_MAIN_TABLES_SPSA_HEA_SCHEDULE_TSV_FIELDS == (
        "hea_spsa_learning_rate",
        "hea_spsa_perturbation",
    )
    assert "family_informed_spsa_eval_repeats" in PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS
    assert "family_informed_spsa_avg_last" in PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS
    assert PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS == (
        "adapt_spsa_a",
        "adapt_spsa_c",
        "adapt_spsa_alpha",
        "adapt_spsa_gamma",
        "adapt_spsa_big_a",
    )
    assert set(PAPER_I_MAIN_TABLES_SPSA_SCHEDULE_TSV_FIELDS).issubset(
        set(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES)
    )
    assert PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_spsa_learning_rate"] == (
        "GENERIC_STATIC_TABLE_HEA_SPSA_LEARNING_RATE"
    )


def test_optimizer_overlay_preserves_blank_prefixed_and_unprefixed_legacy_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_hea_qiskit_vqe as hea

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {"schema": "generic_static_hea_qiskit_vqe_v1", "status": "completed", "rows": [{"status": "ok"}]}

    monkeypatch.setattr(hea, "run_static_hea_qiskit_vqe_single", _fake_runner)
    monkeypatch.setenv("HEA_OPTIMIZER", "SPSA")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_optimizer"], "")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_spsa_learning_rate"], "")

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_hea_qiskit_vqe",
        output_dir=tmp_path / "legacy_defaults",
    )

    assert payload["status"] == "completed"
    assert captured == {"family": "hubbard", "case_id": "hubbard_L2", "output_dir": tmp_path / "legacy_defaults"}


def test_optimizer_overlay_threads_hea_schedule_pair_to_dispatch_stub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_hea_qiskit_vqe as hea

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {"schema": "generic_static_hea_qiskit_vqe_v1", "status": "completed", "rows": [{"status": "ok"}]}

    monkeypatch.setattr(hea, "run_static_hea_qiskit_vqe_single", _fake_runner)
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["optimizer_profile"], PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_spsa_maxiter"], "7")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_spsa_learning_rate"], "0.04")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_spsa_perturbation"], "0.01")

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2_three_model_weak",
        algorithm_id="static_hea_qiskit_vqe",
        output_dir=tmp_path / "hea_profile",
    )

    assert payload["status"] == "completed"
    assert captured["optimizer_profile"] == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
    assert captured["hea_optimizer"] == "spsa"
    assert captured["hea_spsa_maxiter"] == 7
    assert captured["hea_spsa_seed"] == PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS["hea"]["spsa_seed"]
    assert captured["hea_spsa_learning_rate"] == pytest.approx(0.04)
    assert captured["hea_spsa_perturbation"] == pytest.approx(0.01)


def test_optimizer_overlay_threads_family_and_adapt_schedule_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_adapt_variants as variants
    import pipelines.exact_bench.generic_static_family_informed_vqe as family_vqe

    family_captured: dict[str, object] = {}
    adapt_captured: dict[str, object] = {}

    def _fake_family(**kwargs):  # noqa: ANN003, ANN202
        family_captured.update(kwargs)
        return {"schema": "generic_static_family_informed_vqe_v1", "status": "completed", "rows": [{"status": "ok"}]}

    def _fake_adapt(**kwargs):  # noqa: ANN003, ANN202
        adapt_captured.update(kwargs)
        return {"schema": "generic_static_adapt_variants_v4", "status": "completed", "rows": [{"status": "ok"}]}

    monkeypatch.setattr(family_vqe, "run_static_family_informed_vqe_single", _fake_family)
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["optimizer_profile"], PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_seed"], "123")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_a"], "0.05")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_c"], "0.02")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_eval_repeats"], "2")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_avg_last"], "3")

    run_single(
        family="spin_boson",
        case_id="spin_boson_L2_nph1_three_model_weak",
        algorithm_id="static_family_informed_vqe",
        output_dir=tmp_path / "family_profile",
    )

    assert family_captured["family_informed_optimizer"] == "spsa"
    assert family_captured["family_informed_spsa_seed"] == 123
    assert family_captured["family_informed_spsa_a"] == pytest.approx(0.05)
    assert family_captured["family_informed_spsa_c"] == pytest.approx(0.02)
    assert family_captured["family_informed_spsa_eval_repeats"] == 2
    assert family_captured["family_informed_spsa_avg_last"] == 3

    monkeypatch.delenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_seed"])
    monkeypatch.delenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_a"])
    monkeypatch.delenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_c"])
    monkeypatch.delenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_eval_repeats"])
    monkeypatch.delenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_avg_last"])
    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", _fake_adapt)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_maxiter"], "9")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_seed"], "321")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_a"], "0.07")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_c"], "0.03")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_alpha"], "0.602")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_gamma"], "0.101")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_big_a"], "50.0")

    run_single(
        family="hh",
        case_id="hh_L2_nph2_three_model_sym_weak_weak",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "append_profile",
    )

    assert adapt_captured["adapt_optimizer_kind"] == "spsa"
    assert adapt_captured["adapt_spsa_maxiter"] == 9
    assert adapt_captured["adapt_spsa_seed"] == 321
    assert adapt_captured["adapt_spsa_a"] == pytest.approx(0.07)
    assert adapt_captured["adapt_spsa_c"] == pytest.approx(0.03)
    assert adapt_captured["adapt_spsa_alpha"] == pytest.approx(0.602)
    assert adapt_captured["adapt_spsa_gamma"] == pytest.approx(0.101)
    assert adapt_captured["adapt_spsa_big_a"] == pytest.approx(50.0)


def test_optimizer_schedule_fields_fail_closed_for_wrong_dispatch_pair_and_non_spsa(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_a"], "0.05")
    with pytest.raises(ValueError, match="generic-ADAPT optimizer env overlay is only valid"):
        run_single(
            family="hubbard",
            case_id="hubbard_L2_three_model_weak",
            algorithm_id="static_hea_qiskit_vqe",
            output_dir=tmp_path / "wrong_schedule_dispatch",
        )

    monkeypatch.delenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["adapt_spsa_a"])
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_spsa_learning_rate"], "0.04")
    with pytest.raises(ValueError, match="provided together"):
        run_single(
            family="hubbard",
            case_id="hubbard_L2_three_model_weak",
            algorithm_id="static_hea_qiskit_vqe",
            output_dir=tmp_path / "hea_one_sided_schedule",
        )

    monkeypatch.delenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["hea_spsa_learning_rate"])
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_optimizer"], "bfgs")
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES["family_informed_spsa_a"], "0.05")
    with pytest.raises(ValueError, match="require family_informed_optimizer=spsa"):
        run_single(
            family="hubbard",
            case_id="hubbard_L2_three_model_weak",
            algorithm_id="static_family_informed_vqe",
            output_dir=tmp_path / "schedule_requires_spsa",
        )


@pytest.mark.parametrize(
    ("field", "value", "algorithm_id", "family", "case_id", "match"),
    [
        ("hea_spsa_learning_rate", "0", "static_hea_qiskit_vqe", "hubbard", "hubbard_L2_three_model_weak", "positive finite float"),
        ("adapt_spsa_c", "nan", "static_full_meta_append_adapt_vqe", "hh", "hh_L2_nph2_three_model_sym_weak_weak", "positive finite float"),
        (
            "family_informed_spsa_eval_repeats",
            "0",
            "static_family_informed_vqe",
            "hubbard",
            "hubbard_L2_three_model_weak",
            "positive integer",
        ),
    ],
)
def test_optimizer_schedule_env_validates_numeric_bounds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: str,
    algorithm_id: str,
    family: str,
    case_id: str,
    match: str,
) -> None:
    monkeypatch.setenv("TABLE_I_STATIC_SUITE_PROFILE", PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
    monkeypatch.setenv(PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES[field], value)

    with pytest.raises(ValueError, match=match):
        run_single(
            family=family,
            case_id=case_id,
            algorithm_id=algorithm_id,
            output_dir=tmp_path / "bad_schedule_value",
        )
