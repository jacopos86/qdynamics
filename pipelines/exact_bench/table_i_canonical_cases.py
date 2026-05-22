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
from functools import lru_cache
from typing import Mapping

from pipelines.static_adapt.optimization.phase3_policy_optuna import (
    HamiltonianBenchmarkSpec,
    default_static_benchmark_suite,
)

TABLE_I_STANDARD_PROFILE = "standard"
TABLE_I_NPH2_REF3_PROFILE = "nph2_ref3_v1"
TABLE_I_CLEAN_NPH3_REF4_PROFILE = "paper_i_clean_nph3_ref4_v1"
TABLE_I_CLEAN_NPH2_REF3_PROFILE = "paper_i_clean_nph2_ref3_v1"
TABLE_I_CLEAN_NPH2_REF4_PROFILE = "paper_i_clean_nph2_ref4_v1"
TABLE_I_CLEAN_NPH4_REF5_PROFILE = "paper_i_clean_nph4_ref5_v1"
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
TABLE_I_DECLARED_CLEAN_NPH2_REF3_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = _clean_case_ids_by_family(
    n_ph_work=2
)
TABLE_I_DECLARED_CLEAN_NPH2_REF4_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = _clean_case_ids_by_family(
    n_ph_work=2
)
TABLE_I_DECLARED_CLEAN_NPH4_REF5_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = _clean_case_ids_by_family(
    n_ph_work=4
)

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
TABLE_I_EXECUTABLE_CLEAN_NPH2_REF3_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    _clean_executable_case_ids_by_family(n_ph_work=2)
)
TABLE_I_EXECUTABLE_CLEAN_NPH2_REF4_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    _clean_executable_case_ids_by_family(n_ph_work=2)
)
TABLE_I_EXECUTABLE_CLEAN_NPH4_REF5_CASE_IDS_BY_FAMILY: dict[str, tuple[str, ...]] = (
    _clean_executable_case_ids_by_family(n_ph_work=4)
)

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
    TABLE_I_CLEAN_NPH3_REF4_PROFILE: TABLE_I_DECLARED_CLEAN_NPH3_REF4_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH2_REF3_PROFILE: TABLE_I_DECLARED_CLEAN_NPH2_REF3_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH2_REF4_PROFILE: TABLE_I_DECLARED_CLEAN_NPH2_REF4_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH4_REF5_PROFILE: TABLE_I_DECLARED_CLEAN_NPH4_REF5_CASE_IDS_BY_FAMILY,
}
_TABLE_I_EXECUTABLE_CASE_IDS_BY_PROFILE: dict[str, Mapping[str, tuple[str, ...]]] = {
    TABLE_I_STANDARD_PROFILE: TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY,
    TABLE_I_NPH2_REF3_PROFILE: TABLE_I_EXECUTABLE_NPH2_REF3_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH3_REF4_PROFILE: TABLE_I_EXECUTABLE_CLEAN_NPH3_REF4_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH2_REF3_PROFILE: TABLE_I_EXECUTABLE_CLEAN_NPH2_REF3_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH2_REF4_PROFILE: TABLE_I_EXECUTABLE_CLEAN_NPH2_REF4_CASE_IDS_BY_FAMILY,
    TABLE_I_CLEAN_NPH4_REF5_PROFILE: TABLE_I_EXECUTABLE_CLEAN_NPH4_REF5_CASE_IDS_BY_FAMILY,
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
        "clean_nph3": TABLE_I_CLEAN_NPH3_REF4_PROFILE,
        "clean_nph3_ref4": TABLE_I_CLEAN_NPH3_REF4_PROFILE,
        "clean_nph2": TABLE_I_CLEAN_NPH2_REF4_PROFILE,
        "clean_nph2_ref3": TABLE_I_CLEAN_NPH2_REF3_PROFILE,
        "clean_nph2_ref4": TABLE_I_CLEAN_NPH2_REF4_PROFILE,
        "clean_nph4": TABLE_I_CLEAN_NPH4_REF5_PROFILE,
        "clean_nph4_ref5": TABLE_I_CLEAN_NPH4_REF5_PROFILE,
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
    elif profile_key == TABLE_I_CLEAN_NPH4_REF5_PROFILE:
        specs = default_static_benchmark_suite(
            split="train",
            boson_cutoffs=(4,),
            exact_reference_boson_cutoff=5,
            physics_grid_profile="paper_i_clean",
        )
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
    "TABLE_I_CLEAN_NPH2_REF3_PROFILE",
    "TABLE_I_CLEAN_NPH2_REF4_PROFILE",
    "TABLE_I_CLEAN_NPH3_REF4_PROFILE",
    "TABLE_I_CLEAN_NPH4_REF5_PROFILE",
    "TABLE_I_DECLARED_CANONICAL_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_CLEAN_NPH2_REF3_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_CLEAN_NPH2_REF4_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_CLEAN_NPH3_REF4_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_CLEAN_NPH4_REF5_CASE_IDS_BY_FAMILY",
    "TABLE_I_DECLARED_NPH2_REF3_CASE_IDS_BY_FAMILY",
    "TABLE_I_DEFERRED_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CLEAN_NPH2_REF3_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CLEAN_NPH2_REF4_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CLEAN_NPH3_REF4_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CLEAN_NPH4_REF5_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_CASE_IDS_BY_FAMILY",
    "TABLE_I_EXECUTABLE_NPH2_REF3_CASE_IDS_BY_FAMILY",
    "TABLE_I_NPH2_REF3_CASE_IDS_BY_FAMILY",
    "TABLE_I_NPH2_REF3_PROFILE",
    "TABLE_I_STANDARD_PROFILE",
    "TABLE_I_STATIC_SUITE_PROFILE_ENV",
    "table_i_canonical_case_ids",
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
