#!/usr/bin/env python3
"""Paper-I Table-I canonical Hamiltonian case contract.

This module is benchmark glue only.  It separates the declared/future Table-I
case contract from the subset that is executable by the generic static harness
at the current HEAD.  Deferred rows must stay explicit skips until a real
runnable spec and benchmark coverage exist; do not alias a future molecular
vibronic case to a different molecular fixture.
"""

from __future__ import annotations

import os
import math
from dataclasses import replace
from functools import lru_cache
from typing import Mapping

from pipelines.exact_bench.static_benchmark_runtime import (
    HamiltonianBenchmarkSpec,
    ProblemFeatureVector,
    default_static_benchmark_suite,
)
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (
    PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY,
    PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
)

TABLE_I_STANDARD_PROFILE = "standard"
TABLE_I_NPH2_REF3_PROFILE = "nph2_ref3_v1"
TABLE_I_CLEAN_NPH1_REF4_PROFILE = "paper_i_clean_nph1_ref4_v1"
TABLE_I_CLEAN_NPH3_REF4_PROFILE = "paper_i_clean_nph3_ref4_v1"
TABLE_I_CLEAN_NPH2_REF3_PROFILE = "paper_i_clean_nph2_ref3_v1"
TABLE_I_CLEAN_NPH2_REF4_PROFILE = "paper_i_clean_nph2_ref4_v1"
TABLE_I_CLEAN_NPH2_REF5_PROFILE = "paper_i_clean_nph2_ref5_v1"
TABLE_I_CLEAN_NPH4_REF5_PROFILE = "paper_i_clean_nph4_ref5_v1"
TABLE_I_CLEAN_NPH4_REF7_PROFILE = "paper_i_clean_nph4_ref7_v1"
TABLE_I_CLEAN_NPH6_REF9_PROFILE = "paper_i_clean_nph6_ref9_v1"
TABLE_I_CLEAN_H2_NPH3_REF6_PROFILE = "paper_i_clean_h2_nph3_ref6_v1"
TABLE_I_THREE_MODEL_MAIN_PROFILE = "paper_i_three_model_main_20260525_v1"
TABLE_I_THREE_MODEL_REFPLUS2_PROFILE = "paper_i_three_model_refplus2_20260526_v1"
TABLE_I_THREE_MODEL_HH_LOW_WORK_REFPLUS2_PROFILE = "paper_i_three_model_hh_low_work_refplus2_20260526_v1"
TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE = "paper_i_three_model_hh_symmetric_20260527_v1"
TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE = "paper_i_three_model_hh_symmetric_u8_20260611_v1"
TABLE_I_THREE_MODEL_HH_STRESS_GRID_PROFILE = "paper_i_three_model_hh_stress_grid_20260525_v1"
TABLE_I_L3_WEAK_HOLSTEIN_DIAGNOSTIC_PROFILE = "paper_i_l3_weak_holstein_diag_20260624_v1"
TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE = "paper_i_scaling_matrix_20260710_v1"
TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE = (
    "paper_i_higher_l_discriminator_20260719_v1"
)
TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE = (
    "paper_i_hh_completion_samecutoff_nph3_nph7_20260718_v1"
)
TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE = PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
TABLE_I_STATIC_SUITE_PROFILE_ENV = "TABLE_I_STATIC_SUITE_PROFILE"

# Declared Table-I target contract, including future/deferred canonical rows
# that are not executable by the generic static benchmark harness at this HEAD.
TABLE_I_DECLARED_CANONICAL_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hubbard": ("hubbard_L2", "hubbard_L2_u6"),
    "ionic_hubbard": ("ionic_hubbard_L2", "ionic_hubbard_L2_u6"),
    "extended_hubbard": ("extended_hubbard_L2", "extended_hubbard_L2_u6"),
    "ttprime_hubbard": ("ttprime_hubbard_L2", "ttprime_hubbard_L2_u6"),
    "spinless_tv": ("spinless_tv_L2", "spinless_tv_L2_v1p5"),
    "bose_hubbard": ("bose_hubbard_L2", "bose_hubbard_L2_u2"),
    "harmonic_kerr_chain": ("harmonic_kerr_chain_L2", "harmonic_kerr_chain_L2_w0p75"),
    "spin_boson": ("spin_boson_L1", "spin_boson_L1_g0p7"),
    "hh": ("hh_L2",),
    "molecular_vibronic_h2": ("molecular_vibronic_h2_L2",),
}

# Table-I rerun profile requested for the cutoff-aware benchmark sweep.  The
# lattice boson/mixed cases use n_ph_max=2 and reference-cutoff metadata points
# to n_ph_ref=3.  molecular_vibronic_h2 remains declared at its n_ph=1 contract
# but is deferred for generic static Table-I execution at this HEAD.
TABLE_I_DECLARED_NPH2_REF3_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hubbard": ("hubbard_L2", "hubbard_L2_u6"),
    "ionic_hubbard": ("ionic_hubbard_L2", "ionic_hubbard_L2_u6"),
    "extended_hubbard": ("extended_hubbard_L2", "extended_hubbard_L2_u6"),
    "ttprime_hubbard": ("ttprime_hubbard_L2", "ttprime_hubbard_L2_u6"),
    "spinless_tv": ("spinless_tv_L2", "spinless_tv_L2_v1p5"),
    "bose_hubbard": ("bose_hubbard_L2_nph2", "bose_hubbard_L2_nph2_u2"),
    "harmonic_kerr_chain": ("harmonic_kerr_chain_L2_nph2", "harmonic_kerr_chain_L2_nph2_w0p75"),
    "spin_boson": ("spin_boson_L1_nph2", "spin_boson_L1_nph2_g0p7"),
    "hh": ("hh_L2_nph2",),
    "molecular_vibronic_h2": ("molecular_vibronic_h2_L2",),
}


def _clean_case_ids_by_family(*, n_ph_work: int) -> dict[str, tuple[str, ...]]:
    suffix = f"_nph{int(n_ph_work)}"
    return {
        "hubbard": ("hubbard_L2_clean_weak", "hubbard_L2_clean_strong"),
        "ionic_hubbard": ("ionic_hubbard_L2_clean_weak", "ionic_hubbard_L2_clean_strong"),
        "extended_hubbard": ("extended_hubbard_L2_clean_weak", "extended_hubbard_L2_clean_strong"),
        "ttprime_hubbard": ("ttprime_hubbard_L2_clean_weak", "ttprime_hubbard_L2_clean_strong"),
        "spinless_tv": ("spinless_tv_L2_clean_weak", "spinless_tv_L2_clean_strong"),
        "bose_hubbard": (f"bose_hubbard_L2{suffix}_clean_weak", f"bose_hubbard_L2{suffix}_clean_strong"),
        "harmonic_kerr_chain": (
            f"harmonic_kerr_chain_L2{suffix}_clean_weak",
            f"harmonic_kerr_chain_L2{suffix}_clean_strong",
        ),
        "spin_boson": (f"spin_boson_L2{suffix}_clean_weak", f"spin_boson_L2{suffix}_clean_strong"),
        "hh": (f"hh_L2{suffix}_clean_weak", f"hh_L2{suffix}_clean_strong"),
        "molecular_vibronic_h2": (
            "molecular_vibronic_h2_L2_nph1_clean_weak",
            "molecular_vibronic_h2_L2_nph1_clean_strong",
        ),
    }


TABLE_I_DECLARED_CLEAN_NPH3_REF4_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = _clean_case_ids_by_family(
    n_ph_work=3
)
TABLE_I_DECLARED_CLEAN_NPH1_REF4_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = _clean_case_ids_by_family(
    n_ph_work=1
)
TABLE_I_DECLARED_CLEAN_NPH2_REF3_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = _clean_case_ids_by_family(
    n_ph_work=2
)
TABLE_I_DECLARED_CLEAN_NPH2_REF4_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = _clean_case_ids_by_family(
    n_ph_work=2
)
TABLE_I_DECLARED_CLEAN_NPH2_REF5_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = _clean_case_ids_by_family(
    n_ph_work=2
)
TABLE_I_DECLARED_CLEAN_NPH4_REF5_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = _clean_case_ids_by_family(
    n_ph_work=4
)
TABLE_I_DECLARED_CLEAN_NPH4_REF7_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = _clean_case_ids_by_family(
    n_ph_work=4
)
TABLE_I_DECLARED_CLEAN_NPH6_REF9_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = _clean_case_ids_by_family(
    n_ph_work=6
)
TABLE_I_DECLARED_CLEAN_H2_NPH3_REF6_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "molecular_vibronic_h2": ("molecular_vibronic_h2_L2_nph3_clean_strong",),
}
TABLE_I_DECLARED_THREE_MODEL_MAIN_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hubbard": ("hubbard_L2_three_model_weak", "hubbard_L2_three_model_strong"),
    "spin_boson": ("spin_boson_L2_nph1_three_model_weak", "spin_boson_L2_nph2_three_model_strong"),
    "hh": (
        "hh_L2_nph3_three_model_weak_weak",
        "hh_L2_nph2_three_model_strong_weak",
        "hh_L2_nph5_three_model_weak_strong",
        "hh_L2_nph4_three_model_strong_strong",
    ),
}
TABLE_I_DECLARED_THREE_MODEL_REFPLUS2_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    TABLE_I_DECLARED_THREE_MODEL_MAIN_CASE_IDS_BY_FAMILY
)
TABLE_I_DECLARED_THREE_MODEL_HH_LOW_WORK_REFPLUS2_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hh": (
        "hh_L2_nph2_three_model_weak_weak_lowwork",
        "hh_L2_nph1_three_model_strong_weak_lowwork",
        "hh_L2_nph4_three_model_weak_strong_lowwork",
        "hh_L2_nph3_three_model_strong_strong_lowwork",
    ),
}
TABLE_I_DECLARED_THREE_MODEL_HH_SYMMETRIC_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hh": (
        "hh_L2_nph2_three_model_sym_weak_weak",
        "hh_L2_nph2_three_model_sym_strong_weak",
        "hh_L2_nph4_three_model_sym_weak_strong",
        "hh_L2_nph4_three_model_sym_strong_strong",
    ),
}
TABLE_I_DECLARED_THREE_MODEL_HH_SYMMETRIC_U8_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hh": (
        "hh_L2_nph2_three_model_sym_u8_strong_weak",
        "hh_L2_nph4_three_model_sym_u8_strong_strong",
    ),
}
TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS: dict[str, str] = {
    "weak-weak": "hh_L2_nph3_completion_weak_weak",
    "intermediate-weak": "hh_L2_nph3_completion_intermediate_weak",
    "strong-weak": "hh_L2_nph3_completion_strong_weak",
    "weak-strong": "hh_L2_nph7_completion_weak_strong",
    "intermediate-strong": "hh_L2_nph7_completion_intermediate_strong",
    "strong-strong": "hh_L2_nph7_completion_strong_strong",
}
TABLE_I_DECLARED_PAPER_I_HH_COMPLETION_SAME_CUTOFF_CASE_IDS_BY_FAMILY: dict[
    str, tuple[str, ...]
] = {
    "hh": tuple(TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS.values()),
}
TABLE_I_DECLARED_THREE_MODEL_HH_STRESS_GRID_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hh": TABLE_I_DECLARED_THREE_MODEL_MAIN_CASE_IDS_BY_FAMILY["hh"],
}
TABLE_I_DECLARED_PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY
)
TABLE_I_DECLARED_L3_WEAK_HOLSTEIN_DIAGNOSTIC_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hh": (
        "hh_L3_nph1_weak_holstein_weak_weak",
        "hh_L3_nph1_weak_holstein_intermediate_weak",
        "hh_L3_nph1_weak_holstein_strong_weak",
    ),
}
TABLE_I_DECLARED_PAPER_I_SCALING_MATRIX_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hh": tuple(
        f"hh_L{L}_nph{n_ph}_scaling_{regime}"
        for L, n_ph in ((3, 2), (4, 1))
        for regime in (
            "weak_weak",
            "intermediate_weak",
            "strong_weak",
            "weak_strong",
            "intermediate_strong",
            "strong_strong",
        )
    ),
    "hubbard": tuple(
        f"hubbard_L{L}_scaling_{regime}"
        for L in (2, 3, 4)
        for regime in ("weak", "strong")
    ),
    "spin_boson": tuple(
        f"spin_boson_L{L}_nph{n_ph}_scaling_{regime}"
        for L, n_ph in ((2, 4), (3, 3), (3, 2), (4, 1))
        for regime in ("weak", "strong")
    ),
    "bose_hubbard": tuple(
        f"bose_hubbard_L{L}_nph{n_ph}_scaling_{regime}"
        for L, n_ph in ((2, 3), (3, 3), (3, 2), (4, 1))
        for regime in ("weak", "strong")
    ),
}
TABLE_I_DECLARED_PAPER_I_HIGHER_L_DISCRIMINATOR_CASE_IDS_BY_FAMILY: dict[
    str, tuple[str, ...]
] = {
    "hh": (
        "hh_L3_nph3_higher_l_weak_strong",
        "hh_L3_nph3_higher_l_intermediate_strong",
        "hh_L3_nph3_higher_l_strong_strong",
    ),
    "hubbard": (
        "hubbard_L6_higher_l_weak",
        "hubbard_L6_higher_l_strong",
    ),
}

# Generic static Table-I cases executable at this HEAD.  This intentionally
# excludes molecular_vibronic_h2_L2 until the full generic/static Table-I runner
# support is present and tested; it also does not substitute the separate
# molecular_restricted_closed_shell fixture for that future row.
TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hubbard": ("hubbard_L2", "hubbard_L2_u6"),
    "ionic_hubbard": ("ionic_hubbard_L2", "ionic_hubbard_L2_u6"),
    "extended_hubbard": ("extended_hubbard_L2", "extended_hubbard_L2_u6"),
    "ttprime_hubbard": ("ttprime_hubbard_L2", "ttprime_hubbard_L2_u6"),
    "spinless_tv": ("spinless_tv_L2", "spinless_tv_L2_v1p5"),
    "bose_hubbard": ("bose_hubbard_L2", "bose_hubbard_L2_u2"),
    "harmonic_kerr_chain": ("harmonic_kerr_chain_L2", "harmonic_kerr_chain_L2_w0p75"),
    "spin_boson": ("spin_boson_L1", "spin_boson_L1_g0p7"),
    "hh": ("hh_L2",),
}

TABLE_I_EXECUTABLE_NPH2_REF3_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hubbard": ("hubbard_L2", "hubbard_L2_u6"),
    "ionic_hubbard": ("ionic_hubbard_L2", "ionic_hubbard_L2_u6"),
    "extended_hubbard": ("extended_hubbard_L2", "extended_hubbard_L2_u6"),
    "ttprime_hubbard": ("ttprime_hubbard_L2", "ttprime_hubbard_L2_u6"),
    "spinless_tv": ("spinless_tv_L2", "spinless_tv_L2_v1p5"),
    "bose_hubbard": ("bose_hubbard_L2_nph2", "bose_hubbard_L2_nph2_u2"),
    "harmonic_kerr_chain": ("harmonic_kerr_chain_L2_nph2", "harmonic_kerr_chain_L2_nph2_w0p75"),
    "spin_boson": ("spin_boson_L1_nph2", "spin_boson_L1_nph2_g0p7"),
    "hh": ("hh_L2_nph2",),
}


def _clean_executable_case_ids_by_family(*, n_ph_work: int) -> dict[str, tuple[str, ...]]:
    return _clean_case_ids_by_family(n_ph_work=n_ph_work)


TABLE_I_EXECUTABLE_CLEAN_NPH3_REF4_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    _clean_executable_case_ids_by_family(n_ph_work=3)
)
TABLE_I_EXECUTABLE_CLEAN_NPH1_REF4_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    _clean_executable_case_ids_by_family(n_ph_work=1)
)
TABLE_I_EXECUTABLE_CLEAN_NPH2_REF3_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    _clean_executable_case_ids_by_family(n_ph_work=2)
)
TABLE_I_EXECUTABLE_CLEAN_NPH2_REF4_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    _clean_executable_case_ids_by_family(n_ph_work=2)
)
TABLE_I_EXECUTABLE_CLEAN_NPH2_REF5_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    _clean_executable_case_ids_by_family(n_ph_work=2)
)
TABLE_I_EXECUTABLE_CLEAN_NPH4_REF5_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    _clean_executable_case_ids_by_family(n_ph_work=4)
)
TABLE_I_EXECUTABLE_CLEAN_NPH4_REF7_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    _clean_executable_case_ids_by_family(n_ph_work=4)
)
TABLE_I_EXECUTABLE_CLEAN_NPH6_REF9_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    _clean_executable_case_ids_by_family(n_ph_work=6)
)
TABLE_I_EXECUTABLE_CLEAN_H2_NPH3_REF6_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "molecular_vibronic_h2": ("molecular_vibronic_h2_L2_nph3_clean_strong",),
}
TABLE_I_EXECUTABLE_THREE_MODEL_MAIN_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    TABLE_I_DECLARED_THREE_MODEL_MAIN_CASE_IDS_BY_FAMILY
)
TABLE_I_EXECUTABLE_THREE_MODEL_REFPLUS2_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    TABLE_I_DECLARED_THREE_MODEL_REFPLUS2_CASE_IDS_BY_FAMILY
)
TABLE_I_EXECUTABLE_THREE_MODEL_HH_LOW_WORK_REFPLUS2_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    TABLE_I_DECLARED_THREE_MODEL_HH_LOW_WORK_REFPLUS2_CASE_IDS_BY_FAMILY
)
TABLE_I_EXECUTABLE_THREE_MODEL_HH_SYMMETRIC_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    TABLE_I_DECLARED_THREE_MODEL_HH_SYMMETRIC_CASE_IDS_BY_FAMILY
)
TABLE_I_EXECUTABLE_THREE_MODEL_HH_SYMMETRIC_U8_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    TABLE_I_DECLARED_THREE_MODEL_HH_SYMMETRIC_U8_CASE_IDS_BY_FAMILY
)
TABLE_I_EXECUTABLE_PAPER_I_HH_COMPLETION_SAME_CUTOFF_CASE_IDS_BY_FAMILY: dict[
    str, tuple[str, ...]
] = TABLE_I_DECLARED_PAPER_I_HH_COMPLETION_SAME_CUTOFF_CASE_IDS_BY_FAMILY
TABLE_I_EXECUTABLE_THREE_MODEL_HH_STRESS_GRID_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    TABLE_I_DECLARED_THREE_MODEL_HH_STRESS_GRID_CASE_IDS_BY_FAMILY
)
TABLE_I_EXECUTABLE_PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    TABLE_I_DECLARED_PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY
)
TABLE_I_EXECUTABLE_L3_WEAK_HOLSTEIN_DIAGNOSTIC_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    TABLE_I_DECLARED_L3_WEAK_HOLSTEIN_DIAGNOSTIC_CASE_IDS_BY_FAMILY
)
TABLE_I_EXECUTABLE_PAPER_I_SCALING_MATRIX_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    TABLE_I_DECLARED_PAPER_I_SCALING_MATRIX_CASE_IDS_BY_FAMILY
)
TABLE_I_EXECUTABLE_PAPER_I_HIGHER_L_DISCRIMINATOR_CASE_IDS_BY_FAMILY: dict[
    str, tuple[str, ...]
] = TABLE_I_DECLARED_PAPER_I_HIGHER_L_DISCRIMINATOR_CASE_IDS_BY_FAMILY

TABLE_I_DEFERRED_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "molecular_vibronic_h2": ("molecular_vibronic_h2_L2",),
}

_TABLE_I_DEFERRED_CASE_REASONS: dict[tuple[str, str], str] = {
    (
        "molecular_vibronic_h2",
        "molecular_vibronic_h2_L2",
    ): (
        "Table-I molecular_vibronic_h2_L2 is declared for future canonical coverage "
        "but is deferred and not executable at HEAD by the generic static benchmark harness; "
        "do not alias it to molecular_restricted_closed_shell_L2."
    ),
}

# Backwards-compatible runnable aliases.  New code should prefer the explicit
# executable/declared/deferred names above.
TABLE_I_CANONICAL_CASE_IDS_BY_FAMILY = TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY
TABLE_I_NPH2_REF3_CASE_IDS_BY_FAMILY = TABLE_I_EXECUTABLE_NPH2_REF3_CASE_IDS_BY_FAMILY

_TABLE_I_DECLARED_CASE_IDS_BY_PROFILE: dict[str, Mapping[str, tuple[str, ...]]] = {
    TABLE_I_STANDARD_PROFILE: TABLE_I_DECLARED_CANONICAL_CASE_IDS_BY_FAMILY,
    TABLE_I_NPH2_REF3_PROFILE: TABLE_I_DECLARED_NPH2_REF3_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH1_REF4_PROFILE: TABLE_I_DECLARED_CLEAN_NPH1_REF4_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH3_REF4_PROFILE: TABLE_I_DECLARED_CLEAN_NPH3_REF4_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH2_REF3_PROFILE: TABLE_I_DECLARED_CLEAN_NPH2_REF3_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH2_REF4_PROFILE: TABLE_I_DECLARED_CLEAN_NPH2_REF4_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH2_REF5_PROFILE: TABLE_I_DECLARED_CLEAN_NPH2_REF5_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH4_REF5_PROFILE: TABLE_I_DECLARED_CLEAN_NPH4_REF5_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH4_REF7_PROFILE: TABLE_I_DECLARED_CLEAN_NPH4_REF7_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH6_REF9_PROFILE: TABLE_I_DECLARED_CLEAN_NPH6_REF9_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_H2_NPH3_REF6_PROFILE: TABLE_I_DECLARED_CLEAN_H2_NPH3_REF6_CASE_IDS_BY_FAMILY,
    TABLE_I_THREE_MODEL_MAIN_PROFILE: TABLE_I_DECLARED_THREE_MODEL_MAIN_CASE_IDS_BY_FAMILY,
    TABLE_I_THREE_MODEL_REFPLUS2_PROFILE: TABLE_I_DECLARED_THREE_MODEL_REFPLUS2_CASE_IDS_BY_FAMILY,
    TABLE_I_THREE_MODEL_HH_LOW_WORK_REFPLUS2_PROFILE: TABLE_I_DECLARED_THREE_MODEL_HH_LOW_WORK_REFPLUS2_CASE_IDS_BY_FAMILY,
    TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE: TABLE_I_DECLARED_THREE_MODEL_HH_SYMMETRIC_CASE_IDS_BY_FAMILY,
    TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE: TABLE_I_DECLARED_THREE_MODEL_HH_SYMMETRIC_U8_CASE_IDS_BY_FAMILY,
    TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE: TABLE_I_DECLARED_PAPER_I_HH_COMPLETION_SAME_CUTOFF_CASE_IDS_BY_FAMILY,
    TABLE_I_THREE_MODEL_HH_STRESS_GRID_PROFILE: TABLE_I_DECLARED_THREE_MODEL_HH_STRESS_GRID_CASE_IDS_BY_FAMILY,
    TABLE_I_L3_WEAK_HOLSTEIN_DIAGNOSTIC_PROFILE: TABLE_I_DECLARED_L3_WEAK_HOLSTEIN_DIAGNOSTIC_CASE_IDS_BY_FAMILY,
    TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE: TABLE_I_DECLARED_PAPER_I_SCALING_MATRIX_CASE_IDS_BY_FAMILY,
    TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE: TABLE_I_DECLARED_PAPER_I_HIGHER_L_DISCRIMINATOR_CASE_IDS_BY_FAMILY,
    TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE: TABLE_I_DECLARED_PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY,
}
_TABLE_I_EXECUTABLE_CASE_IDS_BY_PROFILE: dict[str, Mapping[str, tuple[str, ...]]] = {
    TABLE_I_STANDARD_PROFILE: TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY,
    TABLE_I_NPH2_REF3_PROFILE: TABLE_I_EXECUTABLE_NPH2_REF3_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH1_REF4_PROFILE: TABLE_I_EXECUTABLE_CLEAN_NPH1_REF4_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH3_REF4_PROFILE: TABLE_I_EXECUTABLE_CLEAN_NPH3_REF4_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH2_REF3_PROFILE: TABLE_I_EXECUTABLE_CLEAN_NPH2_REF3_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH2_REF4_PROFILE: TABLE_I_EXECUTABLE_CLEAN_NPH2_REF4_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH2_REF5_PROFILE: TABLE_I_EXECUTABLE_CLEAN_NPH2_REF5_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH4_REF5_PROFILE: TABLE_I_EXECUTABLE_CLEAN_NPH4_REF5_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH4_REF7_PROFILE: TABLE_I_EXECUTABLE_CLEAN_NPH4_REF7_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH6_REF9_PROFILE: TABLE_I_EXECUTABLE_CLEAN_NPH6_REF9_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_H2_NPH3_REF6_PROFILE: TABLE_I_EXECUTABLE_CLEAN_H2_NPH3_REF6_CASE_IDS_BY_FAMILY,
    TABLE_I_THREE_MODEL_MAIN_PROFILE: TABLE_I_EXECUTABLE_THREE_MODEL_MAIN_CASE_IDS_BY_FAMILY,
    TABLE_I_THREE_MODEL_REFPLUS2_PROFILE: TABLE_I_EXECUTABLE_THREE_MODEL_REFPLUS2_CASE_IDS_BY_FAMILY,
    TABLE_I_THREE_MODEL_HH_LOW_WORK_REFPLUS2_PROFILE: TABLE_I_EXECUTABLE_THREE_MODEL_HH_LOW_WORK_REFPLUS2_CASE_IDS_BY_FAMILY,
    TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE: TABLE_I_EXECUTABLE_THREE_MODEL_HH_SYMMETRIC_CASE_IDS_BY_FAMILY,
    TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE: TABLE_I_EXECUTABLE_THREE_MODEL_HH_SYMMETRIC_U8_CASE_IDS_BY_FAMILY,
    TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE: TABLE_I_EXECUTABLE_PAPER_I_HH_COMPLETION_SAME_CUTOFF_CASE_IDS_BY_FAMILY,
    TABLE_I_THREE_MODEL_HH_STRESS_GRID_PROFILE: TABLE_I_EXECUTABLE_THREE_MODEL_HH_STRESS_GRID_CASE_IDS_BY_FAMILY,
    TABLE_I_L3_WEAK_HOLSTEIN_DIAGNOSTIC_PROFILE: TABLE_I_EXECUTABLE_L3_WEAK_HOLSTEIN_DIAGNOSTIC_CASE_IDS_BY_FAMILY,
    TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE: TABLE_I_EXECUTABLE_PAPER_I_SCALING_MATRIX_CASE_IDS_BY_FAMILY,
    TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE: TABLE_I_EXECUTABLE_PAPER_I_HIGHER_L_DISCRIMINATOR_CASE_IDS_BY_FAMILY,
    TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE: TABLE_I_EXECUTABLE_PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY,
}
# Historical internal name retained for callers that still use canonical as
# runnable-at-HEAD.
_TABLE_I_CASE_IDS_BY_PROFILE = _TABLE_I_EXECUTABLE_CASE_IDS_BY_PROFILE


def table_i_suite_profile(value: str | None = None) -> str:
    """Return the active Table-I static benchmark suite profile."""
    raw = os.environ.get(TABLE_I_STATIC_SUITE_PROFILE_ENV, TABLE_I_STANDARD_PROFILE) if value in {None, ""} else value
    key = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "default": TABLE_I_STANDARD_PROFILE,
        "legacy": TABLE_I_STANDARD_PROFILE,
        "nph2": TABLE_I_NPH2_REF3_PROFILE,
        "nph2_ref3": TABLE_I_NPH2_REF3_PROFILE,
        "paper_i_clean": TABLE_I_CLEAN_NPH3_REF4_PROFILE,
        "clean": TABLE_I_CLEAN_NPH3_REF4_PROFILE,
        "clean_nph1": TABLE_I_CLEAN_NPH1_REF4_PROFILE,
        "clean_nph1_ref4": TABLE_I_CLEAN_NPH1_REF4_PROFILE,
        "clean_nph3": TABLE_I_CLEAN_NPH3_REF4_PROFILE,
        "clean_nph3_ref4": TABLE_I_CLEAN_NPH3_REF4_PROFILE,
        "clean_nph2": TABLE_I_CLEAN_NPH2_REF4_PROFILE,
        "clean_nph2_ref3": TABLE_I_CLEAN_NPH2_REF3_PROFILE,
        "clean_nph2_ref4": TABLE_I_CLEAN_NPH2_REF4_PROFILE,
        "clean_nph2_ref5": TABLE_I_CLEAN_NPH2_REF5_PROFILE,
        "clean_nph4": TABLE_I_CLEAN_NPH4_REF5_PROFILE,
        "clean_nph4_ref5": TABLE_I_CLEAN_NPH4_REF5_PROFILE,
        "clean_nph4_ref7": TABLE_I_CLEAN_NPH4_REF7_PROFILE,
        "clean_nph6": TABLE_I_CLEAN_NPH6_REF9_PROFILE,
        "clean_nph6_ref9": TABLE_I_CLEAN_NPH6_REF9_PROFILE,
        "h2_nph3_ref6": TABLE_I_CLEAN_H2_NPH3_REF6_PROFILE,
        "clean_h2_nph3_ref6": TABLE_I_CLEAN_H2_NPH3_REF6_PROFILE,
        "molecular_vibronic_h2_nph3_ref6": TABLE_I_CLEAN_H2_NPH3_REF6_PROFILE,
        "three_model_main": TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
        "paper_i_three_model_main": TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
        "paper_i_three_model": TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
        "three_model_main_legacy": TABLE_I_THREE_MODEL_MAIN_PROFILE,
        "paper_i_three_model_main_legacy": TABLE_I_THREE_MODEL_MAIN_PROFILE,
        "three_model_refplus2": TABLE_I_THREE_MODEL_REFPLUS2_PROFILE,
        "paper_i_three_model_refplus2": TABLE_I_THREE_MODEL_REFPLUS2_PROFILE,
        "three_model_hh_low_work_refplus2": TABLE_I_THREE_MODEL_HH_LOW_WORK_REFPLUS2_PROFILE,
        "paper_i_three_model_hh_low_work_refplus2": TABLE_I_THREE_MODEL_HH_LOW_WORK_REFPLUS2_PROFILE,
        "three_model_hh_symmetric": TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
        "paper_i_three_model_hh_symmetric": TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
        "hh_symmetric": TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
        "three_model_hh_symmetric_u8": TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE,
        "paper_i_three_model_hh_symmetric_u8": TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE,
        "hh_symmetric_u8": TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE,
        "paper_i_hh_completion_samecutoff": TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE,
        "paper_i_hh_completion_same_cutoff": TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE,
        "hh_completion_samecutoff": TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE,
        "hh_completion_same_cutoff": TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE,
        "three_model_hh_stress_grid": TABLE_I_THREE_MODEL_HH_STRESS_GRID_PROFILE,
        "paper_i_three_model_hh_stress_grid": TABLE_I_THREE_MODEL_HH_STRESS_GRID_PROFILE,
        "l3_weak_holstein_diag": TABLE_I_L3_WEAK_HOLSTEIN_DIAGNOSTIC_PROFILE,
        "paper_i_l3_weak_holstein_diag": TABLE_I_L3_WEAK_HOLSTEIN_DIAGNOSTIC_PROFILE,
        "paper_i_scaling_matrix": TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE,
        "scaling_matrix": TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE,
        "paper_i_higher_l_discriminator": TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE,
        "higher_l_discriminator": TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE,
        "paper_i_main_tables_spsa": TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE,
        "main_tables_spsa": TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE,
        "visible_main_tables_spsa": TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE,
    }
    key = aliases.get(key, key)
    if key not in _TABLE_I_EXECUTABLE_CASE_IDS_BY_PROFILE:
        known = ", ".join(sorted(_TABLE_I_EXECUTABLE_CASE_IDS_BY_PROFILE))
        raise ValueError(f"Unsupported Table-I static suite profile {value!r}; known: {known}")
    return key


def table_i_declared_case_ids_by_family(profile: str | None = None) -> Mapping[str, tuple[str, ...]]:
    """Return declared/future Table-I case IDs for a suite profile."""
    return _TABLE_I_DECLARED_CASE_IDS_BY_PROFILE[table_i_suite_profile(profile)]


def table_i_declared_case_ids(family: str, profile: str | None = None) -> tuple[str, ...]:
    """Return declared/future case IDs for one Table-I family."""
    return tuple(table_i_declared_case_ids_by_family(profile).get(str(family).strip(), ()))


def table_i_executable_case_ids_by_family(profile: str | None = None) -> Mapping[str, tuple[str, ...]]:
    """Return Table-I case IDs executable by the generic static harness at HEAD."""
    return _TABLE_I_EXECUTABLE_CASE_IDS_BY_PROFILE[table_i_suite_profile(profile)]


def table_i_executable_case_ids(family: str, profile: str | None = None) -> tuple[str, ...]:
    """Return executable-at-HEAD case IDs for one Table-I family."""
    return tuple(table_i_executable_case_ids_by_family(profile).get(str(family).strip(), ()))


def table_i_deferred_case_ids_by_family(profile: str | None = None) -> Mapping[str, tuple[str, ...]]:
    """Return declared Table-I cases intentionally deferred at this HEAD."""
    profile_key = table_i_suite_profile(profile)
    declared = table_i_declared_case_ids_by_family(profile_key)
    executable = table_i_executable_case_ids_by_family(profile_key)
    out: dict[str, tuple[str, ...]] = {}
    for family, case_ids in TABLE_I_DEFERRED_CASE_IDS_BY_FAMILY.items():
        deferred = tuple(
            case_id
            for case_id in case_ids
            if case_id in declared.get(family, ()) and case_id not in executable.get(family, ())
        )
        if deferred:
            out[family] = deferred
    return out


def table_i_deferred_case_ids(family: str, profile: str | None = None) -> tuple[str, ...]:
    """Return deferred declared case IDs for one Table-I family."""
    return tuple(table_i_deferred_case_ids_by_family(profile).get(str(family).strip(), ()))


def table_i_deferred_case_reason(family: str, case_id: str, profile: str | None = None) -> str | None:
    """Return a human-readable deferred reason, or ``None`` if the case is not deferred."""
    family_key = str(family).strip()
    case_key = str(case_id).strip()
    if case_key not in table_i_deferred_case_ids(family_key, profile):
        return None
    return _TABLE_I_DEFERRED_CASE_REASONS.get((family_key, case_key), "Table-I case is deferred at this HEAD.")


def table_i_canonical_case_ids_by_family(profile: str | None = None) -> Mapping[str, tuple[str, ...]]:
    """Return the executable-at-HEAD Table-I case-id contract for a suite profile."""
    return table_i_executable_case_ids_by_family(profile)


def table_i_canonical_specs(profile: str | None = None) -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Return executable specs matching the Table-I runnable contract at HEAD."""
    return table_i_executable_specs(profile)


def table_i_executable_specs(profile: str | None = None) -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Return executable-at-HEAD specs matching the Table-I harness contract."""
    return _table_i_executable_specs_cached(table_i_suite_profile(profile))


def _binary_boson_qubits(n_ph_max: int) -> int:
    levels = int(n_ph_max) + 1
    return max(1, int(math.ceil(math.log2(levels))))


def _pipeline_arg_value(args: tuple[str, ...], flag: str) -> str:
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"missing {flag} in base_pipeline_args={args}") from exc


def _static_base_args(
    *,
    family: str,
    L: int,
    u: str,
    g_ep: str,
    n_ph_max: int,
    boundary: str,
    dv: str = "0.0",
    omega0: str = "1.0",
    v_nn: str = "0.0",
    t_prime: str = "0.0",
) -> tuple[str, ...]:
    return (
        "--problem",
        str(family),
        "--L",
        str(int(L)),
        "--t",
        "1.0",
        "--u",
        str(u),
        "--dv",
        str(dv),
        "--omega0",
        str(omega0),
        "--g-ep",
        str(g_ep),
        "--n-ph-max",
        str(int(n_ph_max)),
        "--boson-encoding",
        "binary",
        "--ordering",
        "blocked",
        "--boundary",
        str(boundary),
        "--v-nn",
        str(v_nn),
        "--t-prime",
        str(t_prime),
    )


def _three_model_main_specs(*, reference_offset_override: int | None = None) -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Return the current Paper-I main-table three-Hamiltonian suite.

    The profile reflects the revised manuscript regimes and the per-cell
    phonon cutoffs selected by ED(n) versus the manuscript-specified higher
    reference cutoff at tau=2e-4.
    """

    profile_tag = "physics_profile:paper_i_three_model_main"
    specs: list[HamiltonianBenchmarkSpec] = []
    for regime, u_value, tag in (
        ("weak", "0.5", "u0p5"),
        ("strong", "1.5", "u1p5"),
    ):
        benchmark_id = f"hubbard_L2_three_model_{regime}"
        specs.append(
            HamiltonianBenchmarkSpec(
                benchmark_id=benchmark_id,
                family="hubbard",
                features=ProblemFeatureVector(
                    problem="hubbard",
                    size_label=f"L2_three_model_{regime}",
                    L=2,
                    n_qubits=4,
                    pool_size_hint=64,
                    spinful=True,
                ),
                base_pipeline_args=_static_base_args(
                    family="hubbard",
                    L=2,
                    u=u_value,
                    g_ep="0.5",
                    n_ph_max=1,
                    boundary="periodic",
                ),
                baseline_abs_delta_e=1e-4,
                baseline_count_2q=1000,
                baseline_depth_2q=3000,
                baseline_parameter_count=64,
                baseline_shot_cost_proxy=64,
                exact_reference_n_ph_max=None,
                split="train",
                tags=("static_phase3", "paper_i_three_model_main", profile_tag, regime, tag),
            )
        )

    for regime, g_value, n_ph_work, n_ph_ref, tag in (
        ("weak", "0.05", 1, 5, "g0p05"),
        ("strong", "0.1", 2, 6, "g0p1"),
    ):
        benchmark_id = f"spin_boson_L2_nph{n_ph_work}_three_model_{regime}"
        qpb = _binary_boson_qubits(n_ph_work)
        specs.append(
            HamiltonianBenchmarkSpec(
                benchmark_id=benchmark_id,
                family="spin_boson",
                features=ProblemFeatureVector(
                    problem="spin_boson",
                    size_label=f"L2_nph{n_ph_work}_three_model_{regime}",
                    L=2,
                    n_qubits=2 + 2 * qpb,
                    pool_size_hint=96,
                    bosonic=True,
                ),
                base_pipeline_args=_static_base_args(
                    family="spin_boson",
                    L=2,
                    u="0.0",
                    dv="0.0",
                    g_ep=g_value,
                    n_ph_max=n_ph_work,
                    boundary="open",
                ),
                baseline_abs_delta_e=1e-4,
                baseline_count_2q=1000,
                baseline_depth_2q=3000,
                baseline_parameter_count=64,
                baseline_shot_cost_proxy=64,
                exact_reference_n_ph_max=n_ph_ref,
                split="train",
                tags=(
                    "static_phase3",
                    "paper_i_three_model_main",
                    profile_tag,
                    regime,
                    tag,
                    f"nph{n_ph_work}",
                    f"ref{n_ph_ref}",
                    f"nph{n_ph_work}_ref{n_ph_ref}",
                ),
            )
        )

    hh_points = (
        ("weak_weak", "0.5", "0.5", 3, 6, ("u0p5", "lambda0p5", "g0p5")),
        ("strong_weak", "1.5", "0.5", 2, 5, ("u1p5", "lambda0p5", "g0p5")),
        ("weak_strong", "0.5", "0.8660254037844386", 5, 8, ("u0p5", "lambda1p5", "g_sqrt0p75")),
        ("strong_strong", "1.5", "0.8660254037844386", 4, 7, ("u1p5", "lambda1p5", "g_sqrt0p75")),
    )
    for regime, u_value, g_value, n_ph_work, n_ph_ref, tags in hh_points:
        benchmark_id = f"hh_L2_nph{n_ph_work}_three_model_{regime}"
        qpb = _binary_boson_qubits(n_ph_work)
        specs.append(
            HamiltonianBenchmarkSpec(
                benchmark_id=benchmark_id,
                family="hh",
                features=ProblemFeatureVector(
                    problem="hh",
                    size_label=f"L2_nph{n_ph_work}_three_model_{regime}",
                    L=2,
                    n_qubits=4 + 2 * qpb,
                    pool_size_hint=128,
                    spinful=True,
                    bosonic=True,
                ),
                base_pipeline_args=_static_base_args(
                    family="hh",
                    L=2,
                    u=u_value,
                    g_ep=g_value,
                    n_ph_max=n_ph_work,
                    boundary="open",
                ),
                baseline_abs_delta_e=1e-4,
                baseline_count_2q=1000,
                baseline_depth_2q=3000,
                baseline_parameter_count=64,
                baseline_shot_cost_proxy=64,
                exact_reference_n_ph_max=n_ph_ref,
                split="train",
                tags=(
                    "static_phase3",
                    "paper_i_three_model_main",
                    profile_tag,
                    regime,
                    *tags,
                    f"nph{n_ph_work}",
                    f"ref{n_ph_ref}",
                    f"nph{n_ph_work}_ref{n_ph_ref}",
                ),
            )
        )
    if reference_offset_override is None:
        return tuple(specs)

    offset = int(reference_offset_override)
    if offset < 1:
        raise ValueError("reference_offset_override must be >= 1.")
    adjusted: list[HamiltonianBenchmarkSpec] = []
    for spec in specs:
        if not bool(spec.features.bosonic):
            adjusted.append(spec)
            continue
        n_ph_work = int(_pipeline_arg_value(tuple(spec.base_pipeline_args), "--n-ph-max"))
        n_ph_ref = n_ph_work + offset
        tags = tuple(
            tag
            for tag in spec.tags
            if not (str(tag).startswith("ref") or str(tag).startswith("nph") and "_ref" in str(tag))
        )
        adjusted.append(
            replace(
                spec,
                exact_reference_n_ph_max=n_ph_ref,
                tags=(
                    *tags,
                    f"ref{n_ph_ref}",
                    f"nph{n_ph_work}_ref{n_ph_ref}",
                    f"ref_offset{offset}",
                ),
            )
        )
    return tuple(adjusted)


def _three_model_hh_low_work_refplus2_specs() -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Return the HH-only ED+2 candidate profile with reduced working cutoffs."""

    profile_tag = "physics_profile:paper_i_three_model_hh_low_work_refplus2"
    specs: list[HamiltonianBenchmarkSpec] = []
    hh_points = (
        ("weak_weak", "0.5", "0.5", 2, 4, ("u0p5", "lambda0p5", "g0p5")),
        ("strong_weak", "1.5", "0.5", 1, 3, ("u1p5", "lambda0p5", "g0p5")),
        ("weak_strong", "0.5", "0.8660254037844386", 4, 6, ("u0p5", "lambda1p5", "g_sqrt0p75")),
        ("strong_strong", "1.5", "0.8660254037844386", 3, 5, ("u1p5", "lambda1p5", "g_sqrt0p75")),
    )
    for regime, u_value, g_value, n_ph_work, n_ph_ref, tags in hh_points:
        benchmark_id = f"hh_L2_nph{n_ph_work}_three_model_{regime}_lowwork"
        qpb = _binary_boson_qubits(n_ph_work)
        specs.append(
            HamiltonianBenchmarkSpec(
                benchmark_id=benchmark_id,
                family="hh",
                features=ProblemFeatureVector(
                    problem="hh",
                    size_label=f"L2_nph{n_ph_work}_three_model_{regime}_lowwork",
                    L=2,
                    n_qubits=4 + 2 * qpb,
                    pool_size_hint=128,
                    spinful=True,
                    bosonic=True,
                ),
                base_pipeline_args=_static_base_args(
                    family="hh",
                    L=2,
                    u=u_value,
                    g_ep=g_value,
                    n_ph_max=n_ph_work,
                    boundary="open",
                ),
                baseline_abs_delta_e=1e-4,
                baseline_count_2q=1000,
                baseline_depth_2q=3000,
                baseline_parameter_count=64,
                baseline_shot_cost_proxy=64,
                exact_reference_n_ph_max=n_ph_ref,
                split="train",
                tags=(
                    "static_phase3",
                    "paper_i_three_model_hh_low_work_refplus2",
                    profile_tag,
                    regime,
                    *tags,
                    f"nph{n_ph_work}",
                    f"ref{n_ph_ref}",
                    f"nph{n_ph_work}_ref{n_ph_ref}",
                    "ref_offset2",
                ),
            )
        )
    return tuple(specs)


def _three_model_hh_symmetric_specs() -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Return the current main-body HH symmetric-grid Table-III profile."""

    profile_tag = "physics_profile:paper_i_three_model_hh_symmetric"
    specs: list[HamiltonianBenchmarkSpec] = []
    hh_points = (
        (
            "weak_weak",
            "0.25",
            "0.3535533905932738",
            2,
            5,
            ("u0p25", "lambda0p25", "g_sqrt0p125"),
        ),
        (
            "strong_weak",
            "1.25",
            "0.3535533905932738",
            2,
            5,
            ("u1p25", "lambda0p25", "g_sqrt0p125"),
        ),
        (
            "weak_strong",
            "0.25",
            "0.7905694150420949",
            4,
            7,
            ("u0p25", "lambda1p25", "g_sqrt0p625"),
        ),
        (
            "strong_strong",
            "1.25",
            "0.7905694150420949",
            4,
            7,
            ("u1p25", "lambda1p25", "g_sqrt0p625"),
        ),
    )
    for regime, u_value, g_value, n_ph_work, n_ph_ref, tags in hh_points:
        benchmark_id = f"hh_L2_nph{n_ph_work}_three_model_sym_{regime}"
        qpb = _binary_boson_qubits(n_ph_work)
        specs.append(
            HamiltonianBenchmarkSpec(
                benchmark_id=benchmark_id,
                family="hh",
                features=ProblemFeatureVector(
                    problem="hh",
                    size_label=f"L2_nph{n_ph_work}_three_model_sym_{regime}",
                    L=2,
                    n_qubits=4 + 2 * qpb,
                    pool_size_hint=128,
                    spinful=True,
                    bosonic=True,
                ),
                base_pipeline_args=_static_base_args(
                    family="hh",
                    L=2,
                    u=u_value,
                    g_ep=g_value,
                    n_ph_max=n_ph_work,
                    boundary="open",
                ),
                baseline_abs_delta_e=1e-4,
                baseline_count_2q=1000,
                baseline_depth_2q=3000,
                baseline_parameter_count=64,
                baseline_shot_cost_proxy=64,
                exact_reference_n_ph_max=n_ph_ref,
                split="train",
                tags=(
                    "static_phase3",
                    "paper_i_three_model_hh_symmetric",
                    profile_tag,
                    regime,
                    *tags,
                    f"nph{n_ph_work}",
                    f"ref{n_ph_ref}",
                    f"nph{n_ph_work}_ref{n_ph_ref}",
                    "ref_offset3",
                ),
            )
        )
    return tuple(specs)


def _three_model_hh_symmetric_u8_specs() -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Return the requested HH symmetric strong-Hubbard U/t=8 diagnostic profile."""

    profile_tag = "physics_profile:paper_i_three_model_hh_symmetric_u8"
    specs: list[HamiltonianBenchmarkSpec] = []
    hh_points = (
        (
            "strong_weak",
            "8.0",
            "0.3535533905932738",
            2,
            5,
            ("u8", "lambda0p25", "g_sqrt0p125"),
        ),
        (
            "strong_strong",
            "8.0",
            "0.7905694150420949",
            4,
            7,
            ("u8", "lambda1p25", "g_sqrt0p625"),
        ),
    )
    for regime, u_value, g_value, n_ph_work, n_ph_ref, tags in hh_points:
        benchmark_id = f"hh_L2_nph{n_ph_work}_three_model_sym_u8_{regime}"
        qpb = _binary_boson_qubits(n_ph_work)
        specs.append(
            HamiltonianBenchmarkSpec(
                benchmark_id=benchmark_id,
                family="hh",
                features=ProblemFeatureVector(
                    problem="hh",
                    size_label=f"L2_nph{n_ph_work}_three_model_sym_u8_{regime}",
                    L=2,
                    n_qubits=4 + 2 * qpb,
                    pool_size_hint=128,
                    spinful=True,
                    bosonic=True,
                ),
                base_pipeline_args=_static_base_args(
                    family="hh",
                    L=2,
                    u=u_value,
                    g_ep=g_value,
                    n_ph_max=n_ph_work,
                    boundary="open",
                ),
                baseline_abs_delta_e=1e-4,
                baseline_count_2q=1000,
                baseline_depth_2q=3000,
                baseline_parameter_count=64,
                baseline_shot_cost_proxy=64,
                exact_reference_n_ph_max=n_ph_ref,
                split="train",
                tags=(
                    "static_phase3",
                    "paper_i_three_model_hh_symmetric_u8",
                    profile_tag,
                    regime,
                    *tags,
                    f"nph{n_ph_work}",
                    f"ref{n_ph_ref}",
                    f"nph{n_ph_work}_ref{n_ph_ref}",
                    "ref_offset3",
                ),
            )
        )
    return tuple(specs)


def paper_i_hh_completion_case_id(regime: str) -> str:
    """Resolve one exact completion-grid regime alias fail-closed."""

    key = str(regime).strip().lower().replace("_", "-")
    try:
        return TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS[key]
    except KeyError as exc:
        known = ", ".join(TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS)
        raise ValueError(
            f"Unknown Paper-I HH completion regime alias {regime!r}; known: {known}."
        ) from exc


def _paper_i_hh_completion_same_cutoff_specs() -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Return the six exact L=2 completion-grid HH cases.

    Weak-Holstein cases use n_ph=3 and strong-Holstein cases use n_ph=7.
    The exact reference cutoff equals the working cutoff in every case.
    """

    profile_tag = (
        f"physics_profile:{TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE}"
    )
    points = (
        ("weak-weak", "0.25", "0.353553390593", 3, "u0p25", "lambda0p25"),
        ("intermediate-weak", "1.25", "0.353553390593", 3, "u1p25", "lambda0p25"),
        ("strong-weak", "8.0", "0.353553390593", 3, "u8", "lambda0p25"),
        ("weak-strong", "0.25", "0.790569415042", 7, "u0p25", "lambda1p25"),
        ("intermediate-strong", "1.25", "0.790569415042", 7, "u1p25", "lambda1p25"),
        ("strong-strong", "8.0", "0.790569415042", 7, "u8", "lambda1p25"),
    )
    specs: list[HamiltonianBenchmarkSpec] = []
    for regime_alias, u_value, g_value, n_ph_work, u_tag, lambda_tag in points:
        benchmark_id = paper_i_hh_completion_case_id(regime_alias)
        qpb = _binary_boson_qubits(n_ph_work)
        specs.append(
            HamiltonianBenchmarkSpec(
                benchmark_id=benchmark_id,
                family="hh",
                features=ProblemFeatureVector(
                    problem="hh",
                    size_label=f"L2_nph{n_ph_work}_completion_{regime_alias.replace('-', '_')}",
                    L=2,
                    n_qubits=4 + 2 * qpb,
                    pool_size_hint=128,
                    spinful=True,
                    bosonic=True,
                ),
                base_pipeline_args=_static_base_args(
                    family="hh",
                    L=2,
                    u=u_value,
                    g_ep=g_value,
                    n_ph_max=n_ph_work,
                    boundary="open",
                ),
                baseline_abs_delta_e=1e-4,
                baseline_count_2q=1000,
                baseline_depth_2q=3000,
                baseline_parameter_count=64,
                baseline_shot_cost_proxy=64,
                exact_reference_n_ph_max=n_ph_work,
                split="train",
                tags=(
                    "static_phase3",
                    "paper_i_hh_completion_same_cutoff",
                    profile_tag,
                    regime_alias.replace("-", "_"),
                    f"regime_alias:{regime_alias}",
                    u_tag,
                    lambda_tag,
                    "g_sqrt_lambda_over_2",
                    f"nph{n_ph_work}",
                    "same_cutoff_reference",
                    "working_cutoff_equals_reference_cutoff",
                ),
            )
        )
    expected_ids = tuple(TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS.values())
    observed_ids = tuple(str(spec.benchmark_id) for spec in specs)
    if observed_ids != expected_ids:
        raise ValueError(
            "Paper-I HH completion specs do not match the ordered regime-alias contract: "
            f"{observed_ids!r}!={expected_ids!r}."
        )
    return tuple(specs)


def _l3_weak_holstein_diagnostic_specs() -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Return the user-requested L=3 weak-Holstein diagnostic profile.

    This profile is intentionally narrow: it exists so generic Append/Geo rows
    can resolve the same L=3 weak-Holstein physics points used by the dedicated
    CHTC batch.  It is not a Table-III production profile.
    """

    profile_tag = "physics_profile:paper_i_l3_weak_holstein_diag"
    specs: list[HamiltonianBenchmarkSpec] = []
    hh_points = (
        (
            "weak_weak",
            "0.25",
            "0.3535533905932738",
            ("u0p25", "lambda0p25", "g_sqrt0p125"),
        ),
        (
            "intermediate_weak",
            "1.25",
            "0.3535533905932738",
            ("u1p25", "lambda0p25", "g_sqrt0p125"),
        ),
        (
            "strong_weak",
            "8.0",
            "0.3535533905932738",
            ("u8", "lambda0p25", "g_sqrt0p125"),
        ),
    )
    for regime, u_value, g_value, tags in hh_points:
        n_ph_work = 1
        benchmark_id = f"hh_L3_nph{n_ph_work}_weak_holstein_{regime}"
        qpb = _binary_boson_qubits(n_ph_work)
        specs.append(
            HamiltonianBenchmarkSpec(
                benchmark_id=benchmark_id,
                family="hh",
                features=ProblemFeatureVector(
                    problem="hh",
                    size_label=f"L3_nph{n_ph_work}_weak_holstein_{regime}",
                    L=3,
                    n_qubits=6 + 3 * qpb,
                    pool_size_hint=192,
                    spinful=True,
                    bosonic=True,
                ),
                base_pipeline_args=_static_base_args(
                    family="hh",
                    L=3,
                    u=u_value,
                    g_ep=g_value,
                    n_ph_max=n_ph_work,
                    boundary="open",
                ),
                baseline_abs_delta_e=1e-4,
                baseline_count_2q=1000,
                baseline_depth_2q=3000,
                baseline_parameter_count=96,
                baseline_shot_cost_proxy=96,
                exact_reference_n_ph_max=2,
                split="diagnostic",
                tags=(
                    "static_phase3",
                    "paper_i_l3_weak_holstein_diag",
                    profile_tag,
                    regime,
                    *tags,
                    f"nph{n_ph_work}",
                    "ref2",
                    "nph1_ref2",
                    "diagnostic",
                ),
            )
        )
    return tuple(specs)


def _paper_i_scaling_matrix_specs() -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Return the user-locked Paper-I finite-size/cutoff scaling matrix.

    Bosonic size/cutoff values are ordered pairs, never a Cartesian product.
    All exact-reference metadata is same-cutoff; no unrequested higher-cutoff
    diagnostic is inferred here.
    """

    profile_tag = f"physics_profile:{TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE}"
    specs: list[HamiltonianBenchmarkSpec] = []

    hh_regimes = (
        ("weak_weak", "0.25", "0.3535533905932738", "u0p25", "lambda0p25"),
        ("intermediate_weak", "1.25", "0.3535533905932738", "u1p25", "lambda0p25"),
        ("strong_weak", "8.0", "0.3535533905932738", "u8", "lambda0p25"),
        ("weak_strong", "0.25", "0.7905694150420949", "u0p25", "lambda1p25"),
        ("intermediate_strong", "1.25", "0.7905694150420949", "u1p25", "lambda1p25"),
        ("strong_strong", "8.0", "0.7905694150420949", "u8", "lambda1p25"),
    )
    for L, n_ph_work in ((3, 2), (4, 1)):
        qpb = _binary_boson_qubits(n_ph_work)
        for regime, u_value, g_value, u_tag, lambda_tag in hh_regimes:
            benchmark_id = f"hh_L{L}_nph{n_ph_work}_scaling_{regime}"
            specs.append(
                HamiltonianBenchmarkSpec(
                    benchmark_id=benchmark_id,
                    family="hh",
                    features=ProblemFeatureVector(
                        problem="hh",
                        size_label=f"L{L}_nph{n_ph_work}_scaling_{regime}",
                        L=int(L),
                        n_qubits=int(2 * L + L * qpb),
                        pool_size_hint=int(128 * L),
                        spinful=True,
                        bosonic=True,
                    ),
                    base_pipeline_args=_static_base_args(
                        family="hh",
                        L=int(L),
                        u=u_value,
                        g_ep=g_value,
                        n_ph_max=int(n_ph_work),
                        boundary="open",
                    ),
                    baseline_abs_delta_e=1e-4,
                    baseline_count_2q=1000,
                    baseline_depth_2q=3000,
                    baseline_parameter_count=int(32 * L),
                    baseline_shot_cost_proxy=int(32 * L),
                    exact_reference_n_ph_max=int(n_ph_work),
                    split="train",
                    tags=(
                        "static_phase3",
                        "paper_i_scaling_matrix",
                        profile_tag,
                        regime,
                        u_tag,
                        lambda_tag,
                        f"L{L}",
                        f"nph{n_ph_work}",
                        "same_cutoff_reference",
                    ),
                )
            )

    for L in (2, 3, 4):
        for regime, u_value, u_tag in (("weak", "0.25", "u0p25"), ("strong", "8.0", "u8")):
            benchmark_id = f"hubbard_L{L}_scaling_{regime}"
            specs.append(
                HamiltonianBenchmarkSpec(
                    benchmark_id=benchmark_id,
                    family="hubbard",
                    features=ProblemFeatureVector(
                        problem="hubbard",
                        size_label=f"L{L}_scaling_{regime}",
                        L=int(L),
                        n_qubits=int(2 * L),
                        pool_size_hint=int(32 * L),
                        spinful=True,
                    ),
                    base_pipeline_args=_static_base_args(
                        family="hubbard",
                        L=int(L),
                        u=u_value,
                        g_ep="0.0",
                        n_ph_max=1,
                        boundary="open",
                    ),
                    baseline_abs_delta_e=1e-4,
                    baseline_count_2q=1000,
                    baseline_depth_2q=3000,
                    baseline_parameter_count=int(16 * L),
                    baseline_shot_cost_proxy=int(16 * L),
                    exact_reference_n_ph_max=None,
                    split="train",
                    tags=(
                        "static_phase3",
                        "paper_i_scaling_matrix",
                        profile_tag,
                        regime,
                        u_tag,
                        f"L{L}",
                    ),
                )
            )

    for L, n_ph_work in ((2, 4), (3, 3), (3, 2), (4, 1)):
        qpb = _binary_boson_qubits(n_ph_work)
        for regime, g_value, g_tag in (("weak", "0.05", "g0p05"), ("strong", "0.1", "g0p1")):
            benchmark_id = f"spin_boson_L{L}_nph{n_ph_work}_scaling_{regime}"
            specs.append(
                HamiltonianBenchmarkSpec(
                    benchmark_id=benchmark_id,
                    family="spin_boson",
                    features=ProblemFeatureVector(
                        problem="spin_boson",
                        size_label=f"L{L}_nph{n_ph_work}_scaling_{regime}",
                        L=int(L),
                        n_qubits=int(2 + L * qpb),
                        pool_size_hint=int(48 * L),
                        bosonic=True,
                    ),
                    base_pipeline_args=_static_base_args(
                        family="spin_boson",
                        L=int(L),
                        u="0.0",
                        g_ep=g_value,
                        n_ph_max=int(n_ph_work),
                        boundary="open",
                    ),
                    baseline_abs_delta_e=1e-4,
                    baseline_count_2q=1000,
                    baseline_depth_2q=3000,
                    baseline_parameter_count=int(24 * L),
                    baseline_shot_cost_proxy=int(24 * L),
                    exact_reference_n_ph_max=int(n_ph_work),
                    split="train",
                    tags=(
                        "static_phase3",
                        "paper_i_scaling_matrix",
                        profile_tag,
                        regime,
                        g_tag,
                        f"L{L}",
                        f"nph{n_ph_work}",
                        "same_cutoff_reference",
                    ),
                )
            )

    for L, n_ph_work in ((2, 3), (3, 3), (3, 2), (4, 1)):
        qpb = _binary_boson_qubits(n_ph_work)
        for regime, u_value, u_tag in (("weak", "2.0", "u2"), ("strong", "6.0", "u6")):
            benchmark_id = f"bose_hubbard_L{L}_nph{n_ph_work}_scaling_{regime}"
            specs.append(
                HamiltonianBenchmarkSpec(
                    benchmark_id=benchmark_id,
                    family="bose_hubbard",
                    features=ProblemFeatureVector(
                        problem="bose_hubbard",
                        size_label=f"L{L}_nph{n_ph_work}_scaling_{regime}",
                        L=int(L),
                        n_qubits=int(L * qpb),
                        pool_size_hint=int(48 * L),
                        bosonic=True,
                    ),
                    base_pipeline_args=_static_base_args(
                        family="bose_hubbard",
                        L=int(L),
                        u=u_value,
                        g_ep="0.0",
                        n_ph_max=int(n_ph_work),
                        boundary="open",
                        dv="0.25",
                    ),
                    baseline_abs_delta_e=1e-4,
                    baseline_count_2q=1000,
                    baseline_depth_2q=3000,
                    baseline_parameter_count=int(24 * L),
                    baseline_shot_cost_proxy=int(24 * L),
                    exact_reference_n_ph_max=int(n_ph_work),
                    split="train",
                    tags=(
                        "static_phase3",
                        "paper_i_scaling_matrix",
                        profile_tag,
                        regime,
                        u_tag,
                        f"L{L}",
                        f"nph{n_ph_work}",
                        "same_cutoff_reference",
                    ),
                )
            )

    expected = sum(len(case_ids) for case_ids in TABLE_I_DECLARED_PAPER_I_SCALING_MATRIX_CASE_IDS_BY_FAMILY.values())
    if len(specs) != expected:
        raise ValueError(f"Paper-I scaling matrix generated {len(specs)} specs; expected {expected}.")
    return tuple(specs)


def _paper_i_higher_l_discriminator_specs() -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Return the user-locked five-case higher-L discriminator screen.

    These are explicit physics points, not a Cartesian product.  The HH rows
    use the same binary cutoff for execution and exact-reference reporting.
    """

    profile_tag = f"physics_profile:{TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE}"
    specs: list[HamiltonianBenchmarkSpec] = []
    hh_l = 3
    hh_n_ph = 3
    hh_qpb = _binary_boson_qubits(hh_n_ph)
    for regime, u_value, u_tag in (
        ("weak_strong", "0.25", "u0p25"),
        ("intermediate_strong", "1.25", "u1p25"),
        ("strong_strong", "8.0", "u8"),
    ):
        benchmark_id = f"hh_L3_nph3_higher_l_{regime}"
        specs.append(
            HamiltonianBenchmarkSpec(
                benchmark_id=benchmark_id,
                family="hh",
                features=ProblemFeatureVector(
                    problem="hh",
                    size_label=f"L3_nph3_higher_l_{regime}",
                    L=hh_l,
                    n_qubits=int(2 * hh_l + hh_l * hh_qpb),
                    pool_size_hint=384,
                    spinful=True,
                    bosonic=True,
                ),
                base_pipeline_args=_static_base_args(
                    family="hh",
                    L=hh_l,
                    u=u_value,
                    g_ep="0.7905694150420949",
                    n_ph_max=hh_n_ph,
                    boundary="open",
                ),
                baseline_abs_delta_e=1e-4,
                baseline_count_2q=1000,
                baseline_depth_2q=3000,
                baseline_parameter_count=96,
                baseline_shot_cost_proxy=96,
                exact_reference_n_ph_max=hh_n_ph,
                split="diagnostic",
                tags=(
                    "static_phase3",
                    "paper_i_higher_l_discriminator",
                    profile_tag,
                    regime,
                    u_tag,
                    "lambda1p25",
                    "L3",
                    "nph3",
                    "same_cutoff_reference",
                    "diagnostic",
                ),
            )
        )

    hubbard_l = 6
    for regime, u_value, u_tag in (
        ("weak", "0.25", "u0p25"),
        ("strong", "8.0", "u8"),
    ):
        benchmark_id = f"hubbard_L6_higher_l_{regime}"
        specs.append(
            HamiltonianBenchmarkSpec(
                benchmark_id=benchmark_id,
                family="hubbard",
                features=ProblemFeatureVector(
                    problem="hubbard",
                    size_label=f"L6_higher_l_{regime}",
                    L=hubbard_l,
                    n_qubits=2 * hubbard_l,
                    pool_size_hint=192,
                    spinful=True,
                ),
                base_pipeline_args=_static_base_args(
                    family="hubbard",
                    L=hubbard_l,
                    u=u_value,
                    g_ep="0.0",
                    n_ph_max=1,
                    boundary="open",
                ),
                baseline_abs_delta_e=1e-4,
                baseline_count_2q=1000,
                baseline_depth_2q=3000,
                baseline_parameter_count=96,
                baseline_shot_cost_proxy=96,
                exact_reference_n_ph_max=None,
                split="diagnostic",
                tags=(
                    "static_phase3",
                    "paper_i_higher_l_discriminator",
                    profile_tag,
                    regime,
                    u_tag,
                    "L6",
                    "diagnostic",
                ),
            )
        )

    expected = sum(
        len(case_ids)
        for case_ids in TABLE_I_DECLARED_PAPER_I_HIGHER_L_DISCRIMINATOR_CASE_IDS_BY_FAMILY.values()
    )
    if len(specs) != expected:
        raise ValueError(
            f"Paper-I higher-L discriminator generated {len(specs)} specs; expected {expected}."
        )
    return tuple(specs)


def _paper_i_main_tables_spsa_specs() -> tuple[HamiltonianBenchmarkSpec, ...]:
    """Return the visible Paper-I Tables I--III SPSA-profile case specs."""

    profile_tag = f"optimizer_profile:{TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE}"
    base_specs = tuple(spec for spec in _three_model_main_specs() if str(spec.family) in {"hubbard", "spin_boson"})
    hh_specs = _three_model_hh_symmetric_specs()
    return tuple(replace(spec, tags=(*tuple(spec.tags), profile_tag)) for spec in (*base_specs, *hh_specs))


@lru_cache(maxsize=None)
def _table_i_executable_specs_cached(profile: str) -> tuple[HamiltonianBenchmarkSpec, ...]:
    profile_key = table_i_suite_profile(profile)
    case_ids_by_family = table_i_executable_case_ids_by_family(profile_key)
    wanted = {
        (family, case_id)
        for family, case_ids in case_ids_by_family.items()
        for case_id in case_ids
    }
    if profile_key == TABLE_I_NPH2_REF3_PROFILE:
        specs = default_static_benchmark_suite(
            split="train",
            boson_cutoffs=(1, 2),
            exact_reference_boson_cutoff=3,
            physics_grid_profile="small_robust",
        )
    elif profile_key == TABLE_I_CLEAN_NPH1_REF4_PROFILE:
        specs = default_static_benchmark_suite(
            split="train",
            boson_cutoffs=(1,),
            exact_reference_boson_cutoff=4,
            physics_grid_profile="paper_i_clean",
        )
    elif profile_key == TABLE_I_CLEAN_NPH3_REF4_PROFILE:
        specs = default_static_benchmark_suite(
            split="train",
            boson_cutoffs=(3,),
            exact_reference_boson_cutoff=4,
            physics_grid_profile="paper_i_clean",
        )
    elif profile_key == TABLE_I_CLEAN_NPH2_REF3_PROFILE:
        specs = default_static_benchmark_suite(
            split="train",
            boson_cutoffs=(2,),
            exact_reference_boson_cutoff=3,
            physics_grid_profile="paper_i_clean",
        )
    elif profile_key == TABLE_I_CLEAN_NPH2_REF4_PROFILE:
        specs = default_static_benchmark_suite(
            split="train",
            boson_cutoffs=(2,),
            exact_reference_boson_cutoff=4,
            physics_grid_profile="paper_i_clean",
        )
    elif profile_key == TABLE_I_CLEAN_NPH2_REF5_PROFILE:
        specs = default_static_benchmark_suite(
            split="train",
            boson_cutoffs=(2,),
            exact_reference_boson_cutoff=5,
            physics_grid_profile="paper_i_clean",
        )
    elif profile_key == TABLE_I_CLEAN_NPH4_REF5_PROFILE:
        specs = default_static_benchmark_suite(
            split="train",
            boson_cutoffs=(4,),
            exact_reference_boson_cutoff=5,
            physics_grid_profile="paper_i_clean",
        )
    elif profile_key == TABLE_I_CLEAN_NPH4_REF7_PROFILE:
        specs = default_static_benchmark_suite(
            split="train",
            boson_cutoffs=(4,),
            exact_reference_boson_cutoff=7,
            physics_grid_profile="paper_i_clean",
        )
    elif profile_key == TABLE_I_CLEAN_NPH6_REF9_PROFILE:
        specs = default_static_benchmark_suite(
            split="train",
            boson_cutoffs=(6,),
            exact_reference_boson_cutoff=9,
            physics_grid_profile="paper_i_clean",
        )
    elif profile_key == TABLE_I_CLEAN_H2_NPH3_REF6_PROFILE:
        specs = default_static_benchmark_suite(
            split="train",
            boson_cutoffs=(3,),
            exact_reference_boson_cutoff=6,
            physics_grid_profile="paper_i_clean",
        )
    elif profile_key == TABLE_I_THREE_MODEL_MAIN_PROFILE:
        specs = _three_model_main_specs()
    elif profile_key == TABLE_I_THREE_MODEL_REFPLUS2_PROFILE:
        specs = _three_model_main_specs(reference_offset_override=2)
    elif profile_key == TABLE_I_THREE_MODEL_HH_LOW_WORK_REFPLUS2_PROFILE:
        specs = _three_model_hh_low_work_refplus2_specs()
    elif profile_key == TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE:
        specs = _three_model_hh_symmetric_specs()
    elif profile_key == TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE:
        specs = _three_model_hh_symmetric_u8_specs()
    elif profile_key == TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE:
        specs = _paper_i_hh_completion_same_cutoff_specs()
    elif profile_key == TABLE_I_THREE_MODEL_HH_STRESS_GRID_PROFILE:
        specs = tuple(spec for spec in _three_model_main_specs() if str(spec.family) == "hh")
    elif profile_key == TABLE_I_L3_WEAK_HOLSTEIN_DIAGNOSTIC_PROFILE:
        specs = _l3_weak_holstein_diagnostic_specs()
    elif profile_key == TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE:
        specs = _paper_i_scaling_matrix_specs()
    elif profile_key == TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE:
        specs = _paper_i_higher_l_discriminator_specs()
    elif profile_key == TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE:
        specs = _paper_i_main_tables_spsa_specs()
    else:
        specs = default_static_benchmark_suite(
            split="train",
            boson_cutoff=1,
            exact_reference_boson_cutoff=4,
            physics_grid_profile="small_robust",
        )
    by_key = {(str(spec.family), str(spec.benchmark_id)): spec for spec in specs}
    missing = sorted(wanted - set(by_key))
    if missing:
        raise ValueError(f"Table-I executable specs missing cases for profile={profile_key}: {missing}")
    ordered: list[HamiltonianBenchmarkSpec] = []
    for family, case_ids in case_ids_by_family.items():
        for case_id in case_ids:
            ordered.append(by_key[(family, case_id)])
    return tuple(ordered)


def table_i_canonical_families(profile: str | None = None) -> tuple[str, ...]:
    return tuple(table_i_canonical_case_ids_by_family(profile))


def table_i_canonical_case_ids(family: str, profile: str | None = None) -> tuple[str, ...]:
    return table_i_executable_case_ids(family, profile)


def table_i_canonical_spec_by_case_id(family: str, case_id: str, profile: str | None = None) -> HamiltonianBenchmarkSpec:
    family_key = str(family).strip()
    case_key = str(case_id).strip()
    profile_key = table_i_suite_profile(profile)
    deferred_reason = table_i_deferred_case_reason(family_key, case_key, profile_key)
    if deferred_reason is not None:
        raise ValueError(deferred_reason)
    if case_key not in table_i_executable_case_ids(family_key, profile_key):
        raise ValueError(f"Table-I executable suite profile={profile_key} has no case {family_key}/{case_key}")
    for spec in table_i_executable_specs(profile_key):
        if str(spec.family) == family_key and str(spec.benchmark_id) == case_key:
            return spec
    raise ValueError(f"No executable Table-I spec for profile={profile_key} {family_key}/{case_key}")


__all__ = [
    "TABLE_I_CANONICAL_CASE_IDS_BY_FAMILY",
    "TABLE_I_CLEAN_NPH1_REF4_PROFILE",
    "TABLE_I_CLEAN_NPH2_REF3_PROFILE",
    "TABLE_I_CLEAN_NPH2_REF4_PROFILE",
    "TABLE_I_CLEAN_NPH2_REF5_PROFILE",
    "TABLE_I_CLEAN_NPH3_REF4_PROFILE",
    "TABLE_I_CLEAN_NPH4_REF5_PROFILE",
    "TABLE_I_CLEAN_NPH4_REF7_PROFILE",
    "TABLE_I_CLEAN_NPH6_REF9_PROFILE",
    "TABLE_I_CLEAN_H2_NPH3_REF6_PROFILE",
    "TABLE_I_THREE_MODEL_MAIN_PROFILE",
    "TABLE_I_THREE_MODEL_REFPLUS2_PROFILE",
    "TABLE_I_THREE_MODEL_HH_LOW_WORK_REFPLUS2_PROFILE",
    "TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE",
    "TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE",
    "TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE",
    "TABLE_I_THREE_MODEL_HH_STRESS_GRID_PROFILE",
    "TABLE_I_L3_WEAK_HOLSTEIN_DIAGNOSTIC_PROFILE",
    "TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE",
    "TABLE_I_PAPER_I_HIGHER_L_DISCRIMINATOR_PROFILE",
    "TABLE_I_DECLARED_CANONICAL_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_CLEAN_NPH1_REF4_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_CLEAN_NPH2_REF3_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_CLEAN_NPH2_REF4_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_CLEAN_NPH2_REF5_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_CLEAN_NPH3_REF4_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_CLEAN_NPH4_REF5_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_CLEAN_NPH4_REF7_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_CLEAN_NPH6_REF9_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_CLEAN_H2_NPH3_REF6_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_THREE_MODEL_MAIN_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_THREE_MODEL_REFPLUS2_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_THREE_MODEL_HH_LOW_WORK_REFPLUS2_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_THREE_MODEL_HH_SYMMETRIC_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_THREE_MODEL_HH_SYMMETRIC_U8_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_PAPER_I_HH_COMPLETION_SAME_CUTOFF_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_THREE_MODEL_HH_STRESS_GRID_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_L3_WEAK_HOLSTEIN_DIAGNOSTIC_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_PAPER_I_SCALING_MATRIX_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_PAPER_I_HIGHER_L_DISCRIMINATOR_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_NPH2_REF3_CASE_IDS_BY_FAMILY",
    "TABLE_I_DEFERRED_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CLEAN_NPH1_REF4_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CLEAN_NPH2_REF3_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CLEAN_NPH2_REF4_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CLEAN_NPH2_REF5_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CLEAN_NPH3_REF4_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CLEAN_NPH4_REF5_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CLEAN_NPH4_REF7_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CLEAN_NPH6_REF9_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CLEAN_H2_NPH3_REF6_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_THREE_MODEL_MAIN_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_THREE_MODEL_REFPLUS2_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_THREE_MODEL_HH_LOW_WORK_REFPLUS2_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_THREE_MODEL_HH_SYMMETRIC_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_THREE_MODEL_HH_SYMMETRIC_U8_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_PAPER_I_HH_COMPLETION_SAME_CUTOFF_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_THREE_MODEL_HH_STRESS_GRID_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_L3_WEAK_HOLSTEIN_DIAGNOSTIC_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_PAPER_I_SCALING_MATRIX_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_PAPER_I_HIGHER_L_DISCRIMINATOR_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_PAPER_I_MAIN_TABLES_SPSA_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_NPH2_REF3_CASE_IDS_BY_FAMILY",
    "TABLE_I_NPH2_REF3_CASE_IDS_BY_FAMILY",
    "TABLE_I_NPH2_REF3_PROFILE",
    "TABLE_I_PAPER_I_MAIN_TABLES_SPSA_PROFILE",
    "TABLE_I_STANDARD_PROFILE",
    "TABLE_I_STATIC_SUITE_PROFILE_ENV",
    "table_i_canonical_case_ids",
    "TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS",
    "paper_i_hh_completion_case_id",
    "table_i_canonical_case_ids_by_family",
    "table_i_canonical_families",
    "table_i_canonical_spec_by_case_id",
    "table_i_canonical_specs",
    "table_i_declared_case_ids",
    "table_i_declared_case_ids_by_family",
    "table_i_deferred_case_ids",
    "table_i_deferred_case_ids_by_family",
    "table_i_deferred_case_reason",
    "table_i_executable_case_ids",
    "table_i_executable_case_ids_by_family",
    "table_i_executable_specs",
    "table_i_suite_profile",
]
