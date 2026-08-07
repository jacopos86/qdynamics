#!/usr/bin/env python3
"""Paper-I visible main-table SPSA rerun profile contract.

This module is intentionally pure constants/helpers.  It centralizes the
``paper_i_main_tables_spsa_v1`` literals used by case/profile registration,
record-generation plumbing, and generic dispatch env parsing so the visible
Paper-I Tables I--III SPSA rerun contract does not drift across files.

Blank optimizer TSV/env fields mean legacy/default behavior.  Selecting this
profile is additive and must fail closed if a method-specific optimizer field is
set to a non-SPSA value.
"""

from __future__ import annotations

from typing import Mapping

PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID = "paper_i_main_tables_spsa_v1"
PAPER_I_MAIN_TABLES_SPSA_TARGET = 2e-4
PAPER_I_MAIN_TABLES_SPSA_TARGET_LABEL = "2e-4"

PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hubbard": (
        "hubbard_L2_three_model_weak",
        "hubbard_L2_three_model_strong",
    ),
    "spin_boson": (
        "spin_boson_L2_nph1_three_model_weak",
        "spin_boson_L2_nph2_three_model_strong",
    ),
    "hh": (
        "hh_L2_nph2_three_model_sym_weak_weak",
        "hh_L2_nph2_three_model_sym_strong_weak",
        "hh_L2_nph4_three_model_sym_weak_strong",
        "hh_L2_nph4_three_model_sym_strong_strong",
    ),
}
PAPER_I_MAIN_TABLES_SPSA_CASE_IDS: tuple[str, ...] = tuple(
    case_id
    for case_ids in PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY.values()
    for case_id in case_ids
)
PAPER_I_MAIN_TABLES_SPSA_CASE_KEYS: frozenset[tuple[str, str]] = frozenset(
    (family, case_id)
    for family, case_ids in PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY.items()
    for case_id in case_ids
)

PAPER_I_MAIN_TABLES_SPSA_SMOKE_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hubbard": ("hubbard_L2_three_model_weak",),
    "spin_boson": ("spin_boson_L2_nph1_three_model_weak",),
    "hh": ("hh_L2_nph2_three_model_sym_weak_weak",),
}
PAPER_I_MAIN_TABLES_SPSA_SMOKE_CASE_IDS: tuple[str, ...] = tuple(
    case_id
    for case_ids in PAPER_I_MAIN_TABLES_SPSA_SMOKE_CASE_IDS_BY_FAMILY.values()
    for case_id in case_ids
)

PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS: tuple[str, ...] = (
    "static_hea_qiskit_vqe",
    "static_family_informed_vqe",
    "static_full_meta_append_adapt_vqe",
    "static_qubit_qeb_adapt_vqe",
    "static_tetris_qubit_adapt_vqe",
    "static_geo_adapt_vqe",
)
PAPER_I_MAIN_TABLES_SPSA_SNAKE_ALGORITHM_ID = "static_family_native_adapt_phase3"
PAPER_I_MAIN_TABLES_SPSA_DISPLAYED_ALGORITHM_IDS: tuple[str, ...] = (
    *PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS,
    PAPER_I_MAIN_TABLES_SPSA_SNAKE_ALGORITHM_ID,
)

PAPER_I_MAIN_TABLES_SPSA_PROFILE_ALIASES: dict[str, str | None] = {
    "": None,
    "off": None,
    "none": None,
    "default": None,
    "legacy": None,
    PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID: PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    "paper_i_main_tables_spsa": PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    "main_tables_spsa": PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    "visible_main_tables_spsa": PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
}

PAPER_I_MAIN_TABLES_SPSA_HEA_SCHEDULE_TSV_FIELDS: tuple[str, ...] = (
    "hea_spsa_learning_rate",
    "hea_spsa_perturbation",
)
PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS: tuple[str, ...] = (
    "family_informed_spsa_a",
    "family_informed_spsa_c",
    "family_informed_spsa_alpha",
    "family_informed_spsa_gamma",
    "family_informed_spsa_big_a",
    "family_informed_spsa_eval_repeats",
    "family_informed_spsa_avg_last",
)
PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS: tuple[str, ...] = (
    "adapt_spsa_a",
    "adapt_spsa_c",
    "adapt_spsa_alpha",
    "adapt_spsa_gamma",
    "adapt_spsa_big_a",
)
PAPER_I_MAIN_TABLES_SPSA_SCHEDULE_TSV_FIELDS: tuple[str, ...] = (
    *PAPER_I_MAIN_TABLES_SPSA_HEA_SCHEDULE_TSV_FIELDS,
    *PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS,
    *PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS,
)
PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS: tuple[str, ...] = (
    "optimizer_profile",
    "hea_optimizer",
    "hea_spsa_maxiter",
    "hea_spsa_seed",
    *PAPER_I_MAIN_TABLES_SPSA_HEA_SCHEDULE_TSV_FIELDS,
    "family_informed_optimizer",
    "family_informed_spsa_maxiter",
    "family_informed_spsa_seed",
    *PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS,
    "adapt_optimizer_kind",
    "adapt_spsa_maxiter",
    "adapt_spsa_seed",
    *PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS,
)
PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_PREFIX = "GENERIC_STATIC_TABLE_"
PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES: dict[str, str] = {
    field: PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_PREFIX + field.upper()
    for field in PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS
}

# Full-rerun SPSA budgets deliberately mirror method-native legacy iteration
# budgets where possible.  Schedule constants are additive TSV/env fields:
# blank values preserve method-native legacy/default behavior.
PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS: dict[str, dict[str, int | str]] = {
    "hea": {
        "optimizer": "spsa",
        "spsa_maxiter": 800,
        "spsa_seed": 42,
    },
    "family_informed": {
        "optimizer": "spsa",
        "spsa_maxiter": 250,
        "spsa_seed": 42,
    },
    "adapt": {
        "optimizer_kind": "spsa",
        "spsa_maxiter": 5000,
        "spsa_seed": 42,
    },
}
PAPER_I_MAIN_TABLES_SPSA_SMOKE_BUDGET_DEFAULTS: dict[str, dict[str, int | str]] = {
    "hea": {"optimizer": "spsa", "spsa_maxiter": 1, "spsa_seed": 42},
    "family_informed": {"optimizer": "spsa", "spsa_maxiter": 1, "spsa_seed": 42},
    "adapt": {"optimizer_kind": "spsa", "spsa_maxiter": 1, "spsa_seed": 42},
}


def normalize_paper_i_main_tables_spsa_profile(value: str | None) -> str | None:
    """Normalize optimizer-profile env/TSV input for this SPSA contract."""
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_")
    if key in PAPER_I_MAIN_TABLES_SPSA_PROFILE_ALIASES:
        return PAPER_I_MAIN_TABLES_SPSA_PROFILE_ALIASES[key]
    known = ", ".join(sorted(k for k, v in PAPER_I_MAIN_TABLES_SPSA_PROFILE_ALIASES.items() if v))
    raise ValueError(f"Unsupported optimizer_profile {value!r}; known: {known}")


def paper_i_main_tables_spsa_case_ids_by_family() -> Mapping[str, tuple[str, ...]]:
    """Return the exact visible Paper-I Tables I--III case set for this profile."""
    return PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY


def paper_i_main_tables_spsa_contains_case(family: str, case_id: str) -> bool:
    return (str(family).strip(), str(case_id).strip()) in PAPER_I_MAIN_TABLES_SPSA_CASE_KEYS


__all__ = [
    "PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS",
    "PAPER_I_MAIN_TABLES_SPSA_CASE_IDS",
    "PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY",
    "PAPER_I_MAIN_TABLES_SPSA_CASE_KEYS",
    "PAPER_I_MAIN_TABLES_SPSA_COMPARATOR_ALGORITHM_IDS",
    "PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS",
    "PAPER_I_MAIN_TABLES_SPSA_DISPLAYED_ALGORITHM_IDS",
    "PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS",
    "PAPER_I_MAIN_TABLES_SPSA_HEA_SCHEDULE_TSV_FIELDS",
    "PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES",
    "PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_PREFIX",
    "PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS",
    "PAPER_I_MAIN_TABLES_SPSA_PROFILE_ALIASES",
    "PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID",
    "PAPER_I_MAIN_TABLES_SPSA_SCHEDULE_TSV_FIELDS",
    "PAPER_I_MAIN_TABLES_SPSA_SMOKE_BUDGET_DEFAULTS",
    "PAPER_I_MAIN_TABLES_SPSA_SMOKE_CASE_IDS",
    "PAPER_I_MAIN_TABLES_SPSA_SMOKE_CASE_IDS_BY_FAMILY",
    "PAPER_I_MAIN_TABLES_SPSA_SNAKE_ALGORITHM_ID",
    "PAPER_I_MAIN_TABLES_SPSA_TARGET",
    "PAPER_I_MAIN_TABLES_SPSA_TARGET_LABEL",
    "normalize_paper_i_main_tables_spsa_profile",
    "paper_i_main_tables_spsa_case_ids_by_family",
    "paper_i_main_tables_spsa_contains_case",
]
