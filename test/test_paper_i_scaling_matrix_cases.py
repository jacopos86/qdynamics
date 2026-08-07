from __future__ import annotations

from collections import Counter

from pipelines.exact_bench.table_i_canonical_cases import (
    TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE,
    table_i_executable_case_ids_by_family,
    table_i_executable_specs,
)


EXPECTED_PAIRS = {
    "hh": {(3, 2), (4, 1)},
    "spin_boson": {(2, 4), (3, 3), (3, 2), (4, 1)},
    "bose_hubbard": {(2, 3), (3, 3), (3, 2), (4, 1)},
}


def _arg(spec, flag: str) -> str:
    args = tuple(str(value) for value in spec.base_pipeline_args)
    return args[args.index(flag) + 1]


def test_scaling_matrix_has_exact_user_locked_shape() -> None:
    specs = table_i_executable_specs(TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE)
    assert len(specs) == 34
    assert Counter(spec.family for spec in specs) == {
        "hh": 12,
        "hubbard": 6,
        "spin_boson": 8,
        "bose_hubbard": 8,
    }


def test_scaling_matrix_uses_ordered_size_cutoff_pairs_not_cartesian_products() -> None:
    specs = table_i_executable_specs(TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE)
    for family, expected in EXPECTED_PAIRS.items():
        actual = {
            (int(_arg(spec, "--L")), int(_arg(spec, "--n-ph-max")))
            for spec in specs
            if spec.family == family
        }
        assert actual == expected
    assert (4, 2) not in {
        (int(_arg(spec, "--L")), int(_arg(spec, "--n-ph-max")))
        for spec in specs
        if spec.family == "spin_boson"
    }


def test_scaling_matrix_regime_counts_and_physics_values() -> None:
    specs = table_i_executable_specs(TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE)
    hh = [spec for spec in specs if spec.family == "hh"]
    assert {float(_arg(spec, "--u")) for spec in hh} == {0.25, 1.25, 8.0}
    assert {float(_arg(spec, "--g-ep")) for spec in hh} == {
        0.3535533905932738,
        0.7905694150420949,
    }
    hubbard = [spec for spec in specs if spec.family == "hubbard"]
    assert {float(_arg(spec, "--u")) for spec in hubbard} == {0.25, 8.0}
    spin_boson = [spec for spec in specs if spec.family == "spin_boson"]
    assert {float(_arg(spec, "--g-ep")) for spec in spin_boson} == {0.05, 0.1}
    bose_hubbard = [spec for spec in specs if spec.family == "bose_hubbard"]
    assert {float(_arg(spec, "--u")) for spec in bose_hubbard} == {2.0, 6.0}
    assert {float(_arg(spec, "--dv")) for spec in bose_hubbard} == {0.25}


def test_scaling_matrix_uses_same_cutoff_reference_metadata() -> None:
    specs = table_i_executable_specs(TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE)
    for spec in specs:
        if spec.family == "hubbard":
            assert spec.exact_reference_n_ph_max is None
        else:
            assert spec.exact_reference_n_ph_max == int(_arg(spec, "--n-ph-max"))


def test_scaling_matrix_case_ids_are_executable_for_each_family() -> None:
    case_ids = table_i_executable_case_ids_by_family(TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE)
    assert set(case_ids) == {"hh", "hubbard", "spin_boson", "bose_hubbard"}
    assert sum(len(ids) for ids in case_ids.values()) == 34
