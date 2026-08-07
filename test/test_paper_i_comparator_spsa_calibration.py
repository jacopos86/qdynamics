from __future__ import annotations

import copy
from pathlib import Path

import pytest

from pipelines.exact_bench.paper_i_comparator_spsa_calibration import (
    PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD,
    PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS,
    PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID,
    PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGET_IDS,
    calibration_targets,
    config_sha256_for_path,
    full_method_target_records,
    load_and_validate_config,
    validate_calibration_config,
    validate_full_method_target_records,
    validate_method_id,
)
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (
    PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS,
    paper_i_main_tables_spsa_contains_case,
)

_SMOKE_CONFIG = Path("chtc/phase3_optuna/config/paper_i_comparator_spsa_calibration_v1_smoke.json")


def test_comparator_spsa_calibration_contract_is_closed_visible_matrix() -> None:
    targets = calibration_targets()
    assert PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS == PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS
    assert len(PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS) == 5
    assert PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGET_IDS == (
        "hubbard_family",
        "spin_boson_family",
        "hh_sym_weak_weak",
        "hh_sym_strong_weak",
        "hh_sym_weak_strong",
        "hh_sym_strong_strong",
    )
    assert len(targets) == 6
    for target in targets:
        assert target.case_ids
        assert all(paper_i_main_tables_spsa_contains_case(target.family, case_id) for case_id in target.case_ids)

    records = full_method_target_records()
    assert len(records) == 30
    assert all("case_ids_json" in record for record in records)
    assert {(record["method_id"], record["target_id"]) for record in records} == {
        (method_id, target_id)
        for method_id in PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS
        for target_id in PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGET_IDS
    }
    assert len(validate_full_method_target_records(records)) == 30


@pytest.mark.parametrize(
    "method_id",
    [
        "static_family_native_adapt_phase3",
        "route_a",
        "static_pos_geo_adapt_vqe",
        "static_qiskit_adapt_vqe",
        "static_uccsd_vqe",
        "hh_uccsd_lifted_vqe",
        "static_qse",
        "qse",
        "static_plain_vqe",
        "vqe",
    ],
)
def test_comparator_spsa_calibration_rejects_excluded_methods(method_id: str) -> None:
    with pytest.raises(ValueError, match="excluded|not one of the five retained visible"):
        validate_method_id(method_id)


def test_comparator_spsa_smoke_config_validates_and_hashes_deterministically() -> None:
    validated = load_and_validate_config(_SMOKE_CONFIG)
    expected_hash = config_sha256_for_path(_SMOKE_CONFIG)
    assert validated["profile_id"] == PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID
    assert validated["mode"] == "smoke"
    assert validated["approved_for_full_generation"] is False
    assert validated["config_sha256"] == expected_hash
    assert config_sha256_for_path(_SMOKE_CONFIG) == expected_hash
    assert set(validated["method_maxiter_budgets"]) == set(PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS)
    assert set(validated["per_method_search_spaces"]) == set(PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS)
    for method_id, space in validated["per_method_search_spaces"].items():
        assert set(space).issubset(set(PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD[method_id]))


def test_comparator_spsa_full_config_requires_approval_metadata() -> None:
    config = load_and_validate_config(_SMOKE_CONFIG)
    config.pop("config_sha256", None)
    config["mode"] = "full"

    with pytest.raises(ValueError, match="approved_for_full_generation"):
        validate_calibration_config(config)

    config["approved_for_full_generation"] = True
    with pytest.raises(ValueError, match="approval metadata"):
        validate_calibration_config(config)

    config["approved_by"] = "unit-test-user"
    config["approved_at"] = "2026-05-31T00:00:00Z"
    assert validate_calibration_config(config)["mode"] == "full"


def test_comparator_spsa_full_matrix_validation_uses_case_ids_json() -> None:
    records = [dict(record) for record in full_method_target_records()]
    for record in records:
        record.pop("case_ids", None)
    assert len(validate_full_method_target_records(records)) == 30

    bad = [dict(record) for record in records]
    bad[0]["case_ids_json"] = "not-json"
    with pytest.raises(ValueError, match="malformed case_ids_json"):
        validate_full_method_target_records(bad)

    bad = [dict(record) for record in records]
    bad[0]["case_ids_json"] = "[\"wrong_case\"]"
    with pytest.raises(ValueError, match="case_ids=.*expected"):
        validate_full_method_target_records(bad)


def test_comparator_spsa_config_rejects_wrong_method_schedule_field() -> None:
    config = load_and_validate_config(_SMOKE_CONFIG)
    config.pop("config_sha256", None)
    bad = copy.deepcopy(config)
    bad["per_method_search_spaces"]["static_family_informed_vqe"]["hea_spsa_learning_rate"] = {
        "type": "float",
        "low": 0.01,
        "high": 0.1,
    }

    with pytest.raises(ValueError, match="not an allowed SPSA schedule field"):
        validate_calibration_config(bad)


def test_comparator_spsa_config_rejects_fractional_integer_and_bad_choice_values() -> None:
    config = load_and_validate_config(_SMOKE_CONFIG)
    config.pop("config_sha256", None)

    bad = copy.deepcopy(config)
    bad["n_trials"] = 1.5
    with pytest.raises(ValueError, match="n_trials must be an integer"):
        validate_calibration_config(bad)

    bad = copy.deepcopy(config)
    bad["method_maxiter_budgets"]["static_hea_qiskit_vqe"] = 1.25
    with pytest.raises(ValueError, match="method_maxiter_budgets.static_hea_qiskit_vqe must be an integer"):
        validate_calibration_config(bad)

    bad = copy.deepcopy(config)
    bad["per_method_search_spaces"]["static_family_informed_vqe"]["family_informed_spsa_eval_repeats"] = {
        "type": "choice",
        "choices": [1, 1.5],
    }
    with pytest.raises(ValueError, match=r"choices\[1\] must be an integer"):
        validate_calibration_config(bad)

    bad = copy.deepcopy(config)
    bad["per_method_search_spaces"]["static_hea_qiskit_vqe"]["hea_spsa_learning_rate"] = {
        "type": "choice",
        "choices": ["fast"],
    }
    with pytest.raises(ValueError, match="finite numeric"):
        validate_calibration_config(bad)


def test_comparator_spsa_config_hash_must_be_lowercase_sha256_hex() -> None:
    config = load_and_validate_config(_SMOKE_CONFIG)
    config.pop("config_sha256", None)
    with pytest.raises(ValueError, match="lowercase SHA256 hex"):
        validate_calibration_config(config, config_sha256="Z" * 64)


def test_comparator_spsa_full_matrix_validation_fails_on_missing_record() -> None:
    records = list(full_method_target_records())
    with pytest.raises(ValueError, match="Expected exactly 30"):
        validate_full_method_target_records(records[:-1])
