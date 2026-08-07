#!/usr/bin/env python3
"""Build the compact SR-SNAKE trajectory and Qiskit-cost tracker.

This report intentionally puts trajectories and compiled costs first.  Missing
route/regime evidence is rendered as a blank panel with an explicit status.  A
compact, agent-facing source inventory is kept at the end of the PDF and in the
JSON sidecar.  The script never edits the Paper-I manuscript.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import tarfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import orjson
except ImportError:  # pragma: no cover - repository runtime currently provides it.
    orjson = None


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_RELATIVE_DIR = Path(
    "output/pdf/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715"
)
OUTPUT_DIR = REPO_ROOT / OUTPUT_RELATIVE_DIR
STEM = "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715"
SCHEMA = "paper_i_hh_sr_snake_trajectory_qiskit_tracker_v9"
COMPARATOR_TRACKING_SUMMARY_SCHEMA = "paper_i_hh_comparator_tracking_summary_v1"
COST_ARM_TRACKING_SUMMARY_SCHEMA = "paper_i_hh_cost_arm_tracking_summary_v1"
COST_ARM_REVALIDATION_SCHEMAS = {
    "symmetric": "paper_i_sr_macro_beam_cost_v9_v6_archive_revalidation_v1",
    "one_sided": "paper_i_sr_macro_beam_cost_v10_v6_archive_revalidation_v1",
}
DEFAULT_PLATEAU_JSON = OUTPUT_DIR / "plateau_prefix_costs.json"
PLATEAU_SCHEMA = "paper_i_hh_tracking_plateau_prefix_costs_v1"
PLATEAU_RULE_ID = "first_prefix_within_10pct_of_complete_trajectory_minimum_v1"
DEFAULT_TARGET_JSON = OUTPUT_DIR / "target_energy_prefix_costs.json"
TARGET_SCHEMA = "paper_i_hh_tracking_target_energy_prefix_costs_v1"
TARGET_RULE_ID = "first_prefix_at_or_below_fixed_same_cutoff_error_v1"
TARGET_ABS_ERROR = 2.0e-4
COMPARISON_SCHEMA = "paper_i_hh_top2_sr_vs_append_projected_plateau_v1"
COMPARISON_SELECTION_POLICY = (
    "corrected_complete_six_regime_nph3_7_sr_geomean_plateau_error_v1"
)

CORRECTED_MAIN_ROUTE_DIGEST = (
    "023bc7ac535ee4d88d78dd5336a59dd2fb0543c133fa0a60b009efab75422c91"
)
CORRECTED_FS_PRUNE_ROUTE_DIGEST = (
    "81b072c03f9866817a4fc6173017788223ab8b5ba007d6015315e39d3fb4c30e"
)
PROJECTED_PHASE3_ROUTE_DIGEST = (
    "3ff2abb1455cda3cf8cc2de0cf739172f8cdcfe6b1c9436e1afdd40076cd3ce8"
)
NO_OVERLAP_TRUST_ROUTE_DIGEST = (
    "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2"
)
GUARDED_SINGLETON_ROUTE_DIGEST = (
    "97f89960b93c37d4151a5e6837771c2278c3283fa89ba40671e95fb5806a5e1a"
)
MACRO_PHYSICAL_LANES_ROUTE_DIGEST = (
    "d14d582e532ee41500cd7d3ebaa21b83da91bb3fcf014be53ab8d1049d1452fa"
)
TOP_SR_ROUTE_IDS = (
    "corrected_main_hysteresis_disabled_nph3_7",
    "corrected_fs_prune_hysteresis_disabled_nph3_7",
)
APPEND_PROJECTED_ROUTE_ID = "append_adapt_projected_singleton_nph3_7"
TARGET_COMPARISON_SCHEMA = "paper_i_hh_top2_sr_vs_append_projected_target_energy_v1"
METHOD_REPRESENTATION_COMPARISON_SCHEMA = (
    "paper_i_hh_three_method_representation_target_or_terminal_v1"
)
METHOD_REPRESENTATION_ROUTE_IDS = {
    "macro": (
        "sr_macro_physical_lanes_nph3_7",
        "geo_adapt_macro_nph3_7",
        "append_adapt_macro_nph3_7",
    ),
    "projected_singleton": (
        "sr_guarded_singleton_no_lanes_nph3_7",
        "geo_adapt_projected_singleton_nph3_7",
        "append_adapt_projected_singleton_nph3_7",
    ),
}

REGIMES = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
REGIME_SHORT = {
    "weak_weak": "WW",
    "intermediate_weak": "IW",
    "strong_weak_u8": "SW",
    "weak_strong": "WS",
    "intermediate_strong": "IS",
    "strong_strong_u8": "SS",
}
REGIME_TEX = {
    "weak_weak": "weak--weak",
    "intermediate_weak": "intermediate--weak",
    "strong_weak_u8": "strong--weak",
    "weak_strong": "weak--strong",
    "intermediate_strong": "intermediate--strong",
    "strong_strong_u8": "strong--strong",
}

OLD_OFF_QISKIT = REPO_ROOT / (
    "raw_outputs/paper_i_hh_no_ordinary_novelty_sr_snake_plateau_qiskit_20260717/"
    "comparison/plateau_prefix_qiskit_costs.json"
)
OLD_OFF_RESULTS: dict[str, tuple[str, Path, str | None]] = {
    "weak_weak": (
        "json",
        REPO_ROOT / (
            "raw_outputs/paper_i_hh_sr_snake_weak_weak_undamped_no_prune_no_beam_"
            "no_ordinary_novelty_fallback_on_20260715/json/result.json"
        ),
        None,
    ),
    "intermediate_weak": (
        "json",
        REPO_ROOT / (
            "raw_outputs/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
            "five_20260715_v1_chtc/intermediate_weak/json/result.json"
        ),
        None,
    ),
    "strong_weak_u8": (
        "json",
        REPO_ROOT / (
            "raw_outputs/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
            "strong_weak_u8_r50_repair_20260716_v2_chtc/strong_weak_u8/json/result.json"
        ),
        None,
    ),
    "weak_strong": (
        "tar",
        REPO_ROOT / (
            "raw_outputs/chtc_fetch_paper_i_hh_sr_20260716/"
            "paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
            "r50_continuations_20260715_v1_chtc/weak_strong_transfer.tar.gz"
        ),
        "/weak_strong/json/result.json",
    ),
    "intermediate_strong": (
        "json",
        REPO_ROOT / (
            "raw_outputs/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
            "r50_continuations_20260715_v1_chtc/intermediate_strong/json/result.json"
        ),
        None,
    ),
    "strong_strong_u8": (
        "json",
        REPO_ROOT / (
            "raw_outputs/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
            "r50_continuations_20260715_v1_chtc/strong_strong_u8/json/result.json"
        ),
        None,
    ),
}

OLD_ON_ARCHIVE_DIR = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260716/"
    "paper_i_hh_sr_snake_noprune_nobeam_ordinary_novelty_all_six_20260715_v1_chtc"
)
V4_ARCHIVE_DIR = REPO_ROOT / "raw_outputs/chtc_fetch_paper_i_hh_sr_v4_v6_20260717"
V4_SW_ARCHIVE = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_v4_v7_20260717/"
    "strong_weak_u8_transfer.tar.gz"
)
SYMCOST_ARCHIVE_DIR = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_symcost_20260718/"
    "cluster_8887004_raw_archives"
)
SYMCOST_QISKIT_DIR = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_symcost_20260718/"
    "postprocessed_qiskit"
)

R50_FETCH_ROOT = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_r50_full48_20260719"
)
R50_HEARTBEAT_FETCH = R50_FETCH_ROOT / "heartbeat_20260719T0538Z"
R50_LATE_FETCH = R50_FETCH_ROOT / "heartbeat_20260719T1530Z"
COMPARATOR_LATE_FETCH = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/"
    "retrieval_20260720T131959Z"
)
PROJECTED_PARENT_ANCHOR_ARCHIVE = COMPARATOR_LATE_FETCH / (
    "8900512.0__weak_weak_transfer.tar.gz"
)
PROJECTED_PARENT_ANCHOR_VALIDATION = COMPARATOR_LATE_FETCH / (
    "8900512.0__weak_weak_validation_receipt.json"
)
PROJECTED_PARENT_ANCHOR_AUDIT = REPO_ROOT / (
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_phase3_projected_generalized_parent_anchor_"
    "weak_weak_r50_20260719_v9_chtc/source_locked_sensitivity_audit.json"
)
PROJECTED_PARENT_ANCHOR_AUDIT_SHA256 = (
    "b0554dc08e3a3ffea87f9c7167ad4689822b13c0bc38990a5be2e54887c1542c"
)

PENDING_COST_ARM_NOTES = (
    {
        "schema": "paper_i_hh_tracker_pending_validation_note_v1",
        "route_id": "sr_macro_beam3x2_fs_prune_symmetric_cost_nph3_7",
        "label": "Macro + historical beam 3x2 + live FS-prune, symmetric cost",
        "profile_contract_sha256": (
            "a05ecc8b709db8beac9115d9d0ca39f4faf09e1cbaa10e57bdd674abef9215f0"
        ),
        "cluster": 8900509,
        "status": "pending_fail_closed_local_validation",
        "display_policy": "no scientific route/page until a local pass receipt exists",
    },
    {
        "schema": "paper_i_hh_tracker_pending_validation_note_v1",
        "route_id": "sr_macro_beam3x2_fs_prune_one_sided_cost_nph3_7",
        "label": "Macro + historical beam 3x2 + live FS-prune, one-sided robust cost",
        "profile_contract_sha256": (
            "e3b9f24af40f3572063dd0d13bcca932870505870a8cd7822453b38e01bf6096"
        ),
        "cluster": 8900510,
        "status": "pending_fail_closed_local_validation",
        "display_policy": "no scientific route/page until a local pass receipt exists",
    },
)

COST_ARM_ROUTE_SPECS = (
    {
        "arm": "symmetric",
        "revalidated_subdir": "v9_revalidated",
        "cluster": 8900509,
        "route_id": "sr_macro_beam3x2_fs_prune_symmetric_cost_nph3_7",
        "label": "SR-SNAKE macro + beam 3x2 + FS-prune, symmetric cost",
        "profile_contract_sha256": (
            "a05ecc8b709db8beac9115d9d0ca39f4faf09e1cbaa10e57bdd674abef9215f0"
        ),
        "cost_mode": "family_robust_symmetric_arctan_v1",
        "fallback_policy": "collective_span_novelty_over_symmetric_cost_v1",
    },
    {
        "arm": "one_sided",
        "revalidated_subdir": "v10_revalidated_final",
        "cluster": 8900510,
        "route_id": "sr_macro_beam3x2_fs_prune_one_sided_cost_nph3_7",
        "label": "SR-SNAKE macro + beam 3x2 + FS-prune, one-sided robust cost",
        "profile_contract_sha256": (
            "e3b9f24af40f3572063dd0d13bcca932870505870a8cd7822453b38e01bf6096"
        ),
        "cost_mode": "family_robust_v1",
        "fallback_policy": "collective_span_novelty_over_cost_v1",
    },
)
COST_ARM_ROUTE_IDS = frozenset(
    str(specification["route_id"]) for specification in COST_ARM_ROUTE_SPECS
)


def _cost_arm_summary_path(
    specification: Mapping[str, Any], proc: int, regime: str
) -> Path:
    return COMPARATOR_LATE_FETCH / str(specification["revalidated_subdir"]) / (
        f"{int(specification['cluster'])}.{proc}__{regime}/tracking_summary.json"
    )


def _validated_pair(
    directory: Path, cluster: int, proc: int, regime: str
) -> tuple[Path, Path]:
    prefix = f"{cluster}.{proc}__{regime}"
    return (
        directory / f"{prefix}_transfer.tar.gz",
        directory / f"{prefix}_local_validation_receipt.json",
    )


CORRECTED_MAIN_EVIDENCE = {
    "weak_weak": _validated_pair(
        R50_FETCH_ROOT / "main_sr_8887574_completed_p0_1",
        8887574,
        0,
        "weak_weak",
    ),
    "intermediate_weak": _validated_pair(
        R50_HEARTBEAT_FETCH, 8887574, 1, "intermediate_weak"
    ),
    "strong_weak_u8": _validated_pair(
        R50_HEARTBEAT_FETCH, 8887574, 2, "strong_weak_u8"
    ),
    "weak_strong": _validated_pair(
        R50_LATE_FETCH, 8887574, 3, "weak_strong"
    ),
    "intermediate_strong": _validated_pair(
        R50_LATE_FETCH, 8887574, 4, "intermediate_strong"
    ),
    "strong_strong_u8": _validated_pair(
        R50_LATE_FETCH, 8887574, 5, "strong_strong_u8"
    ),
}

CORRECTED_FS_PRUNE_EVIDENCE = {
    regime: _validated_pair(R50_HEARTBEAT_FETCH, 8887575, proc, regime)
    for proc, regime in enumerate(REGIMES[:3])
}
CORRECTED_FS_PRUNE_EVIDENCE.update(
    {
        regime: _validated_pair(R50_LATE_FETCH, 8887575, proc, regime)
        for proc, regime in enumerate(REGIMES[3:], start=3)
    }
)

PROJECTED_CORE_ARCHIVE_DIR = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/"
    "clusters_8908613_8908614_8908617"
)
PROJECTED_CORE_RESULT_DIR = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/projected_8908614_core/"
    "raw_outputs/paper_i_hh_sr_snake_phase3_projected_generalized_all_six_"
    "r50_20260720_v1_chtc"
)
PROJECTED_LATE_RECEIPT_DIR = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/"
    "projected_8908614_ws_reporting_recovery/compact_artifacts"
)
PROJECTED_LATE_ARCHIVES = {
    "weak_strong": REPO_ROOT / (
        "raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/priority_20260720T2229Z/"
        "8908614.3__weak_strong_transfer.tar.gz"
    ),
    "intermediate_strong": REPO_ROOT / (
        "raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/priority_20260720T2345Z/"
        "8908614.4__intermediate_strong_transfer.tar.gz"
    ),
    "strong_strong_u8": REPO_ROOT / (
        "raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/priority_20260720T2125Z/"
        "8908614.5__strong_strong_u8_transfer.tar.gz"
    ),
}

# Each tuple is (raw transfer archive, fetched validation receipt, optional
# normalized manifest used for the three early core rows whose science finished
# before the stale post-run validator could emit a fetched receipt).
PROJECTED_PHASE3_EVIDENCE: dict[str, tuple[Path, Path | None, Path | None]] = {
    regime: (
        PROJECTED_CORE_ARCHIVE_DIR / f"8908614.{proc}__{regime}_transfer.tar.gz",
        None,
        PROJECTED_CORE_RESULT_DIR / regime / "normalized_run_manifest.json",
    )
    for proc, regime in enumerate(REGIMES[:3])
}
PROJECTED_PHASE3_EVIDENCE.update(
    {
        regime: (
            PROJECTED_LATE_ARCHIVES[regime],
            PROJECTED_LATE_RECEIPT_DIR
            / regime
            / f"{regime}_reporting_recovery_fetched_validation.json",
            None,
        )
        for regime in REGIMES[3:]
    }
)

NO_OVERLAP_FETCH_DIR = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/priority_20260721T0037Z"
)
NO_OVERLAP_RECOVERY_DIR = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/"
    "no_overlap_8958273_weak_cutoff_reporting_recovery/compact_artifacts"
)
NO_OVERLAP_TRUST_EVIDENCE: dict[str, tuple[Path, Path | None, Path | None]] = {
    regime: (
        NO_OVERLAP_FETCH_DIR / f"8958273.{proc}__{regime}_transfer.tar.gz",
        NO_OVERLAP_FETCH_DIR / f"8958273.{proc}__{regime}_validation.json",
        None,
    )
    for proc, regime in enumerate(REGIMES[:3])
}
NO_OVERLAP_TRUST_EVIDENCE["strong_strong_u8"] = (
    REPO_ROOT
    / (
        "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/status_20260721T0149Z/"
        "8958273.5__strong_strong_u8_transfer.tar.gz"
    ),
    NO_OVERLAP_RECOVERY_DIR
    / "strong_strong_u8"
    / "strong_strong_u8_reporting_recovery_fetched_validation.json",
    None,
)
NO_OVERLAP_TRUST_EVIDENCE.update(
    {
        "weak_strong": (
            REPO_ROOT
            / (
                "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/"
                "status_20260721T0732Z/8958273.3__weak_strong_transfer.tar.gz"
            ),
            NO_OVERLAP_RECOVERY_DIR
            / "weak_strong"
            / "weak_strong_reporting_recovery_fetched_validation.json",
            None,
        ),
        "intermediate_strong": (
            REPO_ROOT
            / (
                "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/"
                "status_20260721T0502Z/8958273.4__intermediate_strong_transfer.tar.gz"
            ),
            NO_OVERLAP_RECOVERY_DIR
            / "intermediate_strong"
            / "intermediate_strong_reporting_recovery_fetched_validation.json",
            None,
        ),
    }
)

COMPARATOR_REGIME_FILE = {
    "weak_weak": "weak_weak",
    "intermediate_weak": "intermediate_weak",
    "strong_weak_u8": "strong_weak",
    "weak_strong": "weak_strong",
    "intermediate_strong": "intermediate_strong",
    "strong_strong_u8": "strong_strong",
}


def _comparator_archives(directory: Path, prefix: str) -> dict[str, Path]:
    return {
        regime: directory / f"{prefix}__{file_regime}__r50_transfer.tar.gz"
        for regime, file_regime in COMPARATOR_REGIME_FILE.items()
    }


GEO_MACRO_ARCHIVES = _comparator_archives(
    R50_FETCH_ROOT / "geo_macro_8887540", "geo_macro"
)
APPEND_MACRO_ARCHIVES = _comparator_archives(
    R50_FETCH_ROOT / "append_macro_8887541", "append_macro"
)
APPEND_MACRO_ARCHIVES["intermediate_strong"] = R50_FETCH_ROOT / (
    "append_macro_8887541_completed_p4/"
    "8887541.4__append_macro__intermediate_strong__r50_transfer.tar.gz"
)
APPEND_MACRO_ARCHIVES["strong_strong_u8"] = COMPARATOR_LATE_FETCH / (
    "append_macro__strong_strong__r50_transfer.tar.gz"
)
GEO_PROJECTED_ARCHIVES = _comparator_archives(
    R50_FETCH_ROOT / "comparator_completed_8887546_8887547_p0_2",
    "geo_projected_singleton",
)
GEO_PROJECTED_ARCHIVES["weak_strong"] = COMPARATOR_LATE_FETCH / (
    "geo_projected_singleton__weak_strong__r50_transfer.tar.gz"
)
GEO_PROJECTED_SUCCESSOR_FETCH = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/"
    "geo_projected_singleton_8887762"
)
GEO_PROJECTED_ARCHIVES.update(
    {
        "intermediate_strong": GEO_PROJECTED_SUCCESSOR_FETCH
        / "geo_projected_singleton__intermediate_strong__r50_transfer.tar.gz",
        "strong_strong_u8": GEO_PROJECTED_SUCCESSOR_FETCH
        / "geo_projected_singleton__strong_strong__r50_transfer.tar.gz",
    }
)
APPEND_PROJECTED_ARCHIVES = _comparator_archives(
    R50_FETCH_ROOT / "comparator_completed_8887546_8887547_p0_2",
    "append_projected_singleton",
)
APPEND_PROJECTED_ARCHIVES.update(
    {
        "weak_strong": COMPARATOR_LATE_FETCH
        / "append_projected_singleton__weak_strong__r50_transfer.tar.gz",
        "intermediate_strong": REPO_ROOT
        / (
            "raw_outputs/chtc_fetch_paper_i_hh_append_projected_singleton_20260721/"
            "8887547.4/append_projected_singleton__intermediate_strong__r50_transfer.tar.gz"
        ),
        "strong_strong_u8": COMPARATOR_LATE_FETCH
        / "append_projected_singleton__strong_strong__r50_transfer.tar.gz",
    }
)


def _tracking_summary_path(archive_path: Path) -> Path:
    stem = archive_path.name.removesuffix("_transfer.tar.gz")
    return archive_path.with_name(f"{stem}_tracking_summary.json")


APPEND_MACRO_TRACKING_SUMMARIES = {
    "strong_strong_u8": _tracking_summary_path(
        APPEND_MACRO_ARCHIVES["strong_strong_u8"]
    )
}
GEO_PROJECTED_TRACKING_SUMMARIES = {
    regime: _tracking_summary_path(GEO_PROJECTED_ARCHIVES[regime])
    for regime in ("weak_strong", "intermediate_strong", "strong_strong_u8")
}
APPEND_PROJECTED_TRACKING_SUMMARIES = {
    regime: _tracking_summary_path(APPEND_PROJECTED_ARCHIVES[regime])
    for regime in ("weak_strong", "intermediate_strong", "strong_strong_u8")
}

POOL_COMPLEMENT_FETCH = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_pool_complements_20260719"
)
POOL_STATUS_FETCH = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/status_20260720T0100Z"
)
POOL_HEARTBEAT_FETCH = POOL_COMPLEMENT_FETCH / "heartbeat_20260719T2159Z"
MACRO_RECOVERY_FETCH = POOL_COMPLEMENT_FETCH / (
    "macro_8890778_post_run_recovery_20260719"
)
MACRO_WS_RECOVERY_FETCH = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/"
    "macro_8890778_p3_post_run_recovery_20260720_v3"
)


GUARDED_SINGLETON_EVIDENCE = {
    "weak_weak": _validated_pair(
        POOL_HEARTBEAT_FETCH, 8890777, 0, "weak_weak"
    ),
    "intermediate_weak": _validated_pair(
        POOL_HEARTBEAT_FETCH, 8890777, 1, "intermediate_weak"
    ),
    **{
        regime: (
            POOL_STATUS_FETCH / f"8890777.{proc}__{regime}_transfer.tar.gz",
            POOL_STATUS_FETCH / f"8890777.{proc}__{regime}_validation.json",
        )
        for proc, regime in enumerate(REGIMES[2:], start=2)
    },
}

# The first four macro rows needed a post-run reporting-only recovery because
# the original transfer surface omitted the terminal report archive.  Their
# local validation receipts prove that the recovered result is the unchanged
# completed scientific state.  The final two rows are source-identical
# reporting successors with ordinary transfer archives.
MACRO_PHYSICAL_LANES_EVIDENCE = {
    regime: (
        MACRO_RECOVERY_FETCH
        / f"8890778.{proc}__{regime}_reporting_recovered_v1.tar.gz",
        MACRO_RECOVERY_FETCH / f"8890778.{proc}__{regime}_local_validation.json",
    )
    for proc, regime in enumerate(REGIMES[:3])
}
MACRO_PHYSICAL_LANES_EVIDENCE.update(
    {
        "weak_strong": (
            MACRO_WS_RECOVERY_FETCH
            / "8890778.3__weak_strong_reporting_recovered_v1.tar.gz",
            MACRO_WS_RECOVERY_FETCH
            / "8890778.3__weak_strong_local_validation.json",
        ),
        "intermediate_strong": (
            POOL_STATUS_FETCH
            / "8894440.0__intermediate_strong_transfer.tar.gz",
            POOL_STATUS_FETCH
            / "8894440.0__intermediate_strong_validation.json",
        ),
        "strong_strong_u8": (
            POOL_STATUS_FETCH
            / "8894440.1__strong_strong_u8_transfer.tar.gz",
            POOL_STATUS_FETCH
            / "8894440.1__strong_strong_u8_validation.json",
        ),
    }
)

WHITENED_RESULTS: dict[str, Path] = {
    "weak_weak": REPO_ROOT / (
        "raw_outputs/paper_i_hh_sr_snake_weak_weak_full_accepted_refit_whitened_"
        "20260715/expanded_runtime_projected_logical_v1_r30/json/result.json"
    ),
    "intermediate_weak": REPO_ROOT / (
        "raw_outputs/paper_i_hh_sr_snake_intermediate_weak_full_accepted_refit_"
        "whitened_20260715/expanded_runtime_projected_logical_v1_r30/json/result.json"
    ),
    "strong_weak_u8": REPO_ROOT / (
        "raw_outputs/paper_i_hh_sr_snake_strong_weak_u8_full_accepted_refit_"
        "whitened_20260715/expanded_runtime_projected_logical_v1_r30/json/result.json"
    ),
}
WHITENED_IW_QISKIT = REPO_ROOT / (
    "raw_outputs/paper_i_hh_sr_snake_intermediate_weak_full_accepted_refit_"
    "whitened_20260715/expanded_runtime_projected_logical_v1_r30/qiskit_sidecar.json"
)


@lru_cache(maxsize=None)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _json_loads(raw: bytes) -> Any:
    if orjson is None:
        return json.loads(raw)
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError:
        normalized = raw.replace(b"-Infinity", b"null")
        normalized = normalized.replace(b"Infinity", b"null")
        normalized = normalized.replace(b"NaN", b"null")
        return orjson.loads(normalized)


def _read_json(path: Path) -> Any:
    return _json_loads(path.read_bytes())


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _source_record(path: Path, *, member: str | None = None) -> dict[str, Any]:
    record = {
        "path": _relative(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if member is not None:
        record["member"] = member
    return record


def _latest_source_records(
    sources: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the newest immutable identity for each path/member inventory slot."""

    latest: dict[tuple[str, str | None], dict[str, Any]] = {}
    order: list[tuple[str, str | None]] = []
    for raw in sources:
        record = dict(raw)
        key = (str(record.get("path")), record.get("member"))
        if key not in latest:
            order.append(key)
        latest[key] = record
    return [latest[key] for key in order]


def _source_lock_notes(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and record the exact duplicate source-value anchor once."""

    for path in (
        PROJECTED_PARENT_ANCHOR_ARCHIVE,
        PROJECTED_PARENT_ANCHOR_VALIDATION,
        PROJECTED_PARENT_ANCHOR_AUDIT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(PROJECTED_PARENT_ANCHOR_AUDIT) != PROJECTED_PARENT_ANCHOR_AUDIT_SHA256:
        raise RuntimeError("projected-parent anchor audit SHA-256 drift")

    audit = _read_json(PROJECTED_PARENT_ANCHOR_AUDIT)
    receipt = _read_json(PROJECTED_PARENT_ANCHOR_VALIDATION)
    anchor = audit.get("anchor")
    source = audit.get("source")
    if (
        audit.get("schema") != "source_locked_sensitivity_audit_v1"
        or audit.get("status") != "anchor_pass_fanout_authorized"
        or audit.get("fanout_authorized") is not True
        or not isinstance(anchor, Mapping)
        or not isinstance(source, Mapping)
        or anchor.get("anchor_reproduces_source") is not True
        or anchor.get("non_swept_settings_diff") != []
        or anchor.get("controller_energy_history_exact_match") is not True
        or anchor.get("operator_sequence_match") is not True
        or anchor.get("settings_exact_match") is not True
        or _finite(anchor.get("metric_abs_diff")) != 0.0
        or source.get("route_contract_sha256") != CORRECTED_MAIN_ROUTE_DIGEST
    ):
        raise RuntimeError("projected-parent anchor audit did not prove exact reproduction")

    archive_source = _source_record(PROJECTED_PARENT_ANCHOR_ARCHIVE)
    validation_source = _source_record(PROJECTED_PARENT_ANCHOR_VALIDATION)
    audit_source = _source_record(PROJECTED_PARENT_ANCHOR_AUDIT)
    if (
        anchor.get("anchor_transfer_archive") != archive_source["path"]
        or anchor.get("anchor_transfer_archive_sha256") != archive_source["sha256"]
        or anchor.get("anchor_validation_receipt") != validation_source["path"]
        or anchor.get("anchor_validation_receipt_sha256")
        != validation_source["sha256"]
        or receipt.get("status") != "pass"
        or receipt.get("profile_contract_sha256") != CORRECTED_MAIN_ROUTE_DIGEST
        or receipt.get("result_sha256") != anchor.get("anchor_result_sha256")
        or _integer(receipt.get("target_controller_round")) != 50
    ):
        raise RuntimeError("projected-parent anchor source/receipt identity drift")
    sources.extend((archive_source, validation_source, audit_source))
    return [
        {
            "schema": "paper_i_hh_tracker_source_lock_note_v1",
            "status": "pass",
            "kind": "exact_duplicate_source_value_anchor",
            "regime": "weak_weak",
            "duplicate_of_route_id": "corrected_main_hysteresis_disabled_nph3_7",
            "tested_variable": audit.get("sweep", {}).get("variable"),
            "source_value": anchor.get("value"),
            "terminal_abs_error": anchor.get("terminal_abs_delta_e"),
            "metric_abs_diff": anchor.get("metric_abs_diff"),
            "operator_sequence_match": True,
            "settings_exact_match": True,
            "non_swept_settings_diff": [],
            "display_policy": "manifest_only_no_duplicate_trajectory_or_page",
            "archive_source": archive_source,
            "validation_source": validation_source,
            "audit_source": audit_source,
        }
    ]


def _generated_record(path: Path) -> dict[str, Any]:
    """Hash a staged output while recording its eventual stable target path."""

    return {
        "path": _relative(OUTPUT_DIR / path.name),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _tar_json(path: Path, suffix: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    found: tuple[dict[str, Any], str] | None = None
    with tarfile.open(path, "r|gz") as archive:
        for info in archive:
            if info.name.endswith(suffix):
                if found is not None:
                    raise RuntimeError(
                        f"expected one {suffix!r} member in {path}, observed multiple"
                    )
                member = archive.extractfile(info)
                if member is None:
                    raise RuntimeError(f"cannot extract {info.name} from {path}")
                found = (
                    _json_loads(member.read()),
                    info.name,
                )
            # TarFile retains every TarInfo by default.  These run archives may
            # contain millions of checkpoint members, while this report needs
            # only the named JSON receipts.  Dropping the in-memory index keeps
            # the scan bounded without changing or extracting the archive.
            archive.members.clear()
    if found is None:
        raise RuntimeError(f"expected one {suffix!r} member in {path}, observed none")
    return found


def _tar_json_members(
    path: Path, suffixes: Sequence[str]
) -> dict[str, tuple[dict[str, Any], str]]:
    """Read several JSON members in one sequential compressed-archive pass."""

    if not path.is_file():
        raise FileNotFoundError(path)
    wanted = tuple(suffixes)
    found: dict[str, tuple[dict[str, Any], str]] = {}
    with tarfile.open(path, "r|gz") as archive:
        for info in archive:
            matches = [suffix for suffix in wanted if info.name.endswith(suffix)]
            if matches:
                if len(matches) != 1:
                    raise RuntimeError(
                        f"ambiguous requested suffixes for {info.name!r} in {path}"
                    )
                suffix = matches[0]
                if suffix in found:
                    raise RuntimeError(
                        f"expected one {suffix!r} member in {path}, observed multiple"
                    )
                member = archive.extractfile(info)
                if member is None:
                    raise RuntimeError(f"cannot extract {info.name} from {path}")
                found[suffix] = (
                    _json_loads(member.read()),
                    info.name,
                )
            archive.members.clear()
    missing = [suffix for suffix in wanted if suffix not in found]
    if missing:
        raise RuntimeError(f"missing requested members {missing!r} in {path}")
    return found


def _result_from_source(
    kind: str, path: Path, suffix: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    if kind == "json":
        if not path.is_file():
            raise FileNotFoundError(path)
        return _read_json(path), _source_record(path)
    if kind != "tar" or suffix is None:
        raise ValueError(f"unsupported result source: {(kind, path, suffix)}")
    result, member = _tar_json(path, suffix)
    return result, _source_record(path, member=member)


def _extract_trajectory(payload: Mapping[str, Any]) -> dict[str, Any]:
    adapt = payload.get("adapt_vqe")
    settings = payload.get("settings")
    ground = payload.get("ground_state")
    if not isinstance(adapt, Mapping) or adapt.get("success") is not True:
        raise RuntimeError("result is not a successful ADAPT result")
    if not isinstance(settings, Mapping):
        settings = {}
    if not isinstance(ground, Mapping):
        ground = {}
    history = adapt.get("history")
    if not isinstance(history, list) or not history:
        raise RuntimeError("result has no nonempty ADAPT history")
    exact = _finite(adapt.get("exact_gs_energy"))
    if exact is None:
        exact = _finite(ground.get("exact_energy"))
    points: list[dict[str, Any]] = []
    for index, row in enumerate(history, start=1):
        if not isinstance(row, Mapping):
            continue
        # Controller/history position is the reporting prefix identity.  Active
        # depth can fall after pruning and must never replace the x coordinate.
        round_id = _integer(row.get("outer_iteration")) or index
        error = _finite(row.get("delta_abs_current"))
        energy = _finite(row.get("energy_after_opt"))
        if error is None and energy is not None and exact is not None:
            error = abs(energy - exact)
        if error is None:
            continue
        points.append(
            {
                "round": round_id,
                "error": error,
                "active_depth": _integer(row.get("depth")),
            }
        )
    if not points:
        raise RuntimeError("result history has no finite same-cutoff error trajectory")
    terminal_error = _finite(adapt.get("abs_delta_e"))
    if terminal_error is None:
        terminal_error = points[-1]["error"]
    estimator_accounting = adapt.get("estimator_call_accounting")
    all_branch_work = (
        estimator_accounting.get("all_branch_search_work")
        if isinstance(estimator_accounting, Mapping)
        else None
    )
    s_alg = (
        _integer(all_branch_work.get("S_alg"))
        if isinstance(all_branch_work, Mapping)
        else None
    )
    return {
        "status": "complete",
        "n_ph": _integer(settings.get("n_ph_max")),
        "rounds": len(history),
        "active_depth": _integer(adapt.get("ansatz_depth")),
        "terminal_error": terminal_error,
        "s_alg": s_alg,
        "s_alg_scope": "all_branch_search_work" if s_alg is not None else None,
        "trajectory": points,
    }


def _empty_result(status: str = "pending") -> dict[str, Any]:
    return {
        "status": status,
        "n_ph": None,
        "rounds": None,
        "active_depth": None,
        "terminal_error": None,
        "s_alg": None,
        "s_alg_scope": None,
        "trajectory": [],
        "source": None,
    }


def _normalize_regime_filename(path: Path) -> str:
    stem = path.name.removesuffix("_transfer.tar.gz")
    if stem not in REGIMES:
        raise RuntimeError(f"unrecognized regime archive name: {path.name}")
    return stem


def _qiskit_metrics(payload: Mapping[str, Any]) -> dict[str, int | None]:
    current = payload.get("current_jr_fake_marrakesh_convention")
    if isinstance(current, Mapping) and isinstance(current.get("metrics"), Mapping):
        metrics = current["metrics"]
        return {
            "N2q": _integer(metrics.get("N2q")),
            "D2q": _integer(metrics.get("D2q")),
            "Dc": _integer(metrics.get("Dc")),
        }
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        return {
            "N2q": _integer(metrics.get("N2q")),
            "D2q": _integer(metrics.get("D2q")),
            "Dc": _integer(metrics.get("Dc")),
        }
    return {
        "N2q": _integer(payload.get("compiled_count_2q_total")),
        "D2q": _integer(payload.get("compiled_depth_2q_total")),
        "Dc": _integer(payload.get("compiled_depth_total")),
    }


def _build_whitened_route(sources: list[dict[str, Any]]) -> dict[str, Any]:
    results = {regime: _empty_result("not run") for regime in REGIMES}
    for regime, path in WHITENED_RESULTS.items():
        payload = _read_json(path)
        result = _extract_trajectory(payload)
        source = _source_record(path)
        result["source"] = source
        sources.append(source)
        results[regime] = result
    costs = {regime: None for regime in REGIMES}
    if WHITENED_IW_QISKIT.is_file():
        payload = _read_json(WHITENED_IW_QISKIT)
        costs["intermediate_weak"] = _qiskit_metrics(payload)
        sources.append(_source_record(WHITENED_IW_QISKIT))
    return {
        "id": "accepted_refit_whitened_weak_holstein_anchors",
        "label": "Accepted-refit whitening anchors",
        "subtitle": "expanded-runtime/projected-logical chart; weak-Holstein n_ph=2",
        "policy": "historical selector; supported-FS full accepted refit; diagnostic anchors",
        "cost_convention": "Paper-I basis-gate transpile where available",
        "results": results,
        "costs": costs,
    }


def _build_old_off_route(sources: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    for regime in REGIMES:
        payload, source = _result_from_source(*OLD_OFF_RESULTS[regime])
        result = _extract_trajectory(payload)
        result["source"] = source
        sources.append(source)
        results[regime] = result
    qiskit_payload = _read_json(OLD_OFF_QISKIT)
    sources.append(_source_record(OLD_OFF_QISKIT))
    costs = {regime: None for regime in REGIMES}
    for row in qiskit_payload.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        regime = str(row.get("regime") or "")
        if regime in costs and isinstance(row.get("qiskit"), Mapping):
            costs[regime] = {
                "N2q": _integer(row["qiskit"].get("N2q")),
                "D2q": _integer(row["qiskit"].get("D2q")),
                "Dc": _integer(row["qiskit"].get("Dc")),
            }
    return {
        "id": "legacy_no_ordinary_novelty_nph2_4",
        "label": "Novelty OFF control",
        "subtitle": "no beam, no prune, undamped; n_ph=2 weak / n_ph=4 strong",
        "policy": "ordinary Phase-II/III novelty off; conditional fallback retained",
        "cost_convention": "Paper-I basis-gate transpile at selected displayed endpoint",
        "results": results,
        "costs": costs,
    }


def _build_archive_route(
    *,
    route_id: str,
    label: str,
    subtitle: str,
    policy: str,
    archive_dir: Path,
    sources: list[dict[str, Any]],
    include_qiskit: bool,
    qiskit_sidecar_dir: Path | None = None,
    archive_overrides: Mapping[str, Path] | None = None,
    cost_convention: str,
) -> dict[str, Any]:
    archives = {
        _normalize_regime_filename(path): path
        for path in sorted(archive_dir.glob("*_transfer.tar.gz"))
    }
    if archive_overrides:
        archives.update(archive_overrides)
    results = {regime: _empty_result("pending") for regime in REGIMES}
    costs = {regime: None for regime in REGIMES}
    for regime, path in archives.items():
        payload, member = _tar_json(path, f"/{regime}/json/result.json")
        result = _extract_trajectory(payload)
        source = _source_record(path, member=member)
        result["source"] = source
        results[regime] = result
        sources.append(source)
        if include_qiskit:
            sidecar, sidecar_member = _tar_json(path, f"/{regime}/qiskit_cost_sidecar.json")
            costs[regime] = _qiskit_metrics(sidecar)
            sources.append(_source_record(path, member=sidecar_member))
        if qiskit_sidecar_dir is not None:
            sidecar_path = qiskit_sidecar_dir / regime / "qiskit_cost_sidecar.json"
            if sidecar_path.is_file():
                costs[regime] = _qiskit_metrics(_read_json(sidecar_path))
                sources.append(_source_record(sidecar_path))
    return {
        "id": route_id,
        "label": label,
        "subtitle": subtitle,
        "policy": policy,
        "cost_convention": cost_convention,
        "results": results,
        "costs": costs,
    }


def _build_validated_archive_route(
    *,
    route_id: str,
    label: str,
    subtitle: str,
    policy: str,
    evidence: Mapping[str, tuple[Path, Path]],
    expected_route_digest: str,
    sources: list[dict[str, Any]],
    cost_convention: str,
) -> dict[str, Any]:
    """Ingest only archive/receipt pairs that passed the source-locked validator."""

    results = {regime: _empty_result("awaiting local validation") for regime in REGIMES}
    costs = {regime: None for regime in REGIMES}
    for regime in REGIMES:
        archive_path, receipt_path = evidence[regime]
        if not archive_path.is_file() and not receipt_path.is_file():
            continue
        if not archive_path.is_file() or not receipt_path.is_file():
            raise RuntimeError(
                f"incomplete validated evidence pair for {route_id}/{regime}: "
                f"archive={archive_path.is_file()} receipt={receipt_path.is_file()}"
            )
        receipt = _read_json(receipt_path)
        if receipt.get("status") != "pass":
            raise RuntimeError(f"validation receipt did not pass: {receipt_path}")
        if receipt.get("profile_contract_sha256") != expected_route_digest:
            raise RuntimeError(
                f"route digest drift for {route_id}/{regime}: "
                f"{receipt.get('profile_contract_sha256')}"
            )
        if _integer(receipt.get("target_controller_round")) != 50:
            raise RuntimeError(f"round-50 receipt required: {receipt_path}")
        evidence_validation = receipt.get("scientific_evidence_validation")
        if not isinstance(evidence_validation, Mapping):
            raise RuntimeError(f"missing scientific validation payload: {receipt_path}")
        if evidence_validation.get("supported_rank_recorded_each_round") is not True:
            raise RuntimeError(f"incomplete Phase-III supported-rank evidence: {receipt_path}")
        ledger = evidence_validation.get("active_prefix_estimator_ledger_receipts")
        if not isinstance(ledger, Mapping) or ledger.get("closure_passed") is not True:
            raise RuntimeError(f"estimator ledger did not close: {receipt_path}")

        payload, member = _tar_json(archive_path, f"/{regime}/json/result.json")
        result = _extract_trajectory(payload)
        if _integer(result.get("rounds")) != 50:
            raise RuntimeError(f"result did not contain 50 completed rounds: {archive_path}")
        archive_source = _source_record(archive_path, member=member)
        receipt_source = _source_record(receipt_path)
        result["source"] = archive_source
        result["validation_receipt"] = receipt_source
        results[regime] = result
        sources.extend((archive_source, receipt_source))

        metrics = receipt.get("current_fake_marrakesh_metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"validated Qiskit metrics are missing: {receipt_path}")
        costs[regime] = _qiskit_metrics({"metrics": metrics})
        del payload, receipt
        gc.collect()

    return {
        "id": route_id,
        "label": label,
        "subtitle": subtitle,
        "policy": policy,
        "cost_convention": cost_convention,
        "route_contract_sha256": expected_route_digest,
        "results": results,
        "costs": costs,
    }


def _validate_priority_sr_payload(
    payload: Mapping[str, Any],
    *,
    regime: str,
    expected_route_digest: str,
    expected_trust_policy: str,
) -> None:
    """Fail closed on the scientific identity displayed by the new SR pages."""

    settings = payload.get("settings")
    adapt = payload.get("adapt_vqe")
    if not isinstance(settings, Mapping) or not isinstance(adapt, Mapping):
        raise RuntimeError(f"priority SR result lacks settings/adapt payload: {regime}")
    if settings.get("sr_route_profile_contract_sha256") != expected_route_digest:
        raise RuntimeError(f"priority SR route digest drift: {regime}")
    expected_n_ph = 3 if regime in REGIMES[:3] else 7
    if _integer(settings.get("n_ph_max")) != expected_n_ph:
        raise RuntimeError(f"priority SR cutoff drift: {regime}")
    if (
        settings.get("historical_singleton_coordinate_solve_policy")
        != "supported_metric_projected_generalized_trust_v1"
    ):
        raise RuntimeError(f"priority SR Phase-III solve drift: {regime}")
    if (
        settings.get("historical_singleton_trust_region_update_policy")
        != expected_trust_policy
    ):
        raise RuntimeError(f"priority SR trust-update drift: {regime}")
    if settings.get("phase3_response_coordinate_scope") != "full_active_plus_singleton_v1":
        raise RuntimeError(f"priority SR response scope drift: {regime}")
    contract = settings.get("sr_route_profile_contract")
    invariants = contract.get("semantic_invariants") if isinstance(contract, Mapping) else None
    if not isinstance(invariants, Mapping):
        raise RuntimeError(f"priority SR semantic invariants missing: {regime}")
    required = {
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
    for key, expected in required.items():
        if invariants.get(key) != expected:
            raise RuntimeError(f"priority SR invariant drift {key}: {regime}")
    accepted_refit = adapt.get("accepted_refit")
    if (
        not isinstance(accepted_refit, Mapping)
        or accepted_refit.get("supported_fs_whitened") is not True
        or accepted_refit.get("full_ansatz") is not True
    ):
        raise RuntimeError(f"priority SR accepted-refit drift: {regime}")


def _validate_priority_sr_receipt(
    receipt: Mapping[str, Any],
    *,
    regime: str,
    expected_route_digest: str,
    require_no_overlap: bool,
) -> dict[str, int | None]:
    if receipt.get("status") != "pass":
        raise RuntimeError(f"priority SR validation did not pass: {regime}")
    if receipt.get("profile_contract_sha256") != expected_route_digest:
        raise RuntimeError(f"priority SR receipt route drift: {regime}")
    if _integer(receipt.get("target_controller_round")) != 50:
        raise RuntimeError(f"priority SR receipt is not round 50: {regime}")
    scientific = receipt.get("scientific_evidence_validation")
    if not isinstance(scientific, Mapping):
        raise RuntimeError(f"priority SR scientific receipt missing: {regime}")
    if scientific.get("supported_rank_recorded_each_round") is not True:
        raise RuntimeError(f"priority SR supported-rank receipt incomplete: {regime}")
    ledger = scientific.get("active_prefix_estimator_ledger_receipts")
    if not isinstance(ledger, Mapping) or ledger.get("closure_passed") is not True:
        raise RuntimeError(f"priority SR estimator ledger did not close: {regime}")
    projected = receipt.get("projected_generalized_phase3_validation")
    if (
        not isinstance(projected, Mapping)
        or projected.get("status") != "pass"
        or projected.get("supported_metric_whitening_active") is not False
        or projected.get("accepted_powell_refit_whitening_active") is not True
        or _integer(projected.get("controller_rounds")) != 50
    ):
        raise RuntimeError(f"priority SR projected-Phase-III receipt drift: {regime}")
    if require_no_overlap:
        trust = receipt.get("no_overlap_trust_validation")
        if (
            not isinstance(trust, Mapping)
            or trust.get("status") != "pass"
            or _integer(trust.get("controller_rounds")) != 50
            or _integer(trust.get("endpoint_overlap_measurement_count")) != 0
            or _integer(trust.get("endpoint_overlap_query_charge")) != 0
            or trust.get("accepted_powell_refit_whitening_active") is not True
        ):
            raise RuntimeError(f"priority SR no-overlap receipt drift: {regime}")
    metrics = receipt.get("current_fake_marrakesh_metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f"priority SR Qiskit receipt missing: {regime}")
    return _qiskit_metrics({"metrics": metrics})


def _build_priority_sr_route(
    *,
    route_id: str,
    label: str,
    subtitle: str,
    policy: str,
    evidence: Mapping[str, tuple[Path, Path | None, Path | None]],
    expected_route_digest: str,
    expected_trust_policy: str,
    require_no_overlap: bool,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    results = {regime: _empty_result("pending on CHTC") for regime in REGIMES}
    costs = {regime: None for regime in REGIMES}
    for regime in REGIMES:
        pair = evidence.get(regime)
        if pair is None:
            continue
        archive_path, receipt_path, manifest_path = pair
        if not archive_path.is_file():
            if receipt_path is None or not receipt_path.is_file():
                continue
            raise RuntimeError(f"priority SR receipt exists without archive: {regime}")
        payload, member = _tar_json(archive_path, f"/{regime}/json/result.json")
        _validate_priority_sr_payload(
            payload,
            regime=regime,
            expected_route_digest=expected_route_digest,
            expected_trust_policy=expected_trust_policy,
        )
        result = _extract_trajectory(payload)
        if _integer(result.get("rounds")) != 50:
            raise RuntimeError(f"priority SR result is not 50 rounds: {regime}")
        source = _source_record(archive_path, member=member)
        result["source"] = source
        result["phase3_support_projection_active"] = True
        result["phase3_support_whitening_active"] = False
        result["accepted_refit_whitening_active"] = True
        sources.append(source)
        if receipt_path is not None:
            if not receipt_path.is_file():
                raise RuntimeError(f"priority SR fetched receipt missing: {receipt_path}")
            receipt = _read_json(receipt_path)
            costs[regime] = _validate_priority_sr_receipt(
                receipt,
                regime=regime,
                expected_route_digest=expected_route_digest,
                require_no_overlap=require_no_overlap,
            )
            receipt_source = _source_record(receipt_path)
            result["validation_receipt"] = receipt_source
            result["validation_mode"] = "fetched_fail_closed_receipt"
            sources.append(receipt_source)
        elif manifest_path is not None:
            if require_no_overlap:
                raise RuntimeError(f"no-overlap route requires a fetched receipt: {regime}")
            if not manifest_path.is_file():
                raise RuntimeError(f"priority SR normalized manifest missing: {manifest_path}")
            manifest = _read_json(manifest_path)
            manifest_settings = manifest.get("settings")
            if not isinstance(manifest_settings, Mapping):
                manifest_settings = manifest.get("normalized_settings")
            if isinstance(manifest_settings, Mapping):
                manifest_digest = manifest_settings.get("sr_route_profile_contract_sha256")
                if manifest_digest not in (None, expected_route_digest):
                    raise RuntimeError(f"priority SR manifest route drift: {regime}")
            manifest_source = _source_record(manifest_path)
            result["route_identity_receipt"] = manifest_source
            result["validation_mode"] = "embedded_route_contract_plus_normalized_manifest"
            sources.append(manifest_source)
        else:
            raise RuntimeError(f"priority SR row lacks validation provenance: {regime}")
        results[regime] = result
        del payload
        gc.collect()
    return {
        "id": route_id,
        "label": label,
        "subtitle": subtitle,
        "policy": policy,
        "cost_convention": "validated JR FakeMarrakesh, optimization level 1 where fetched receipts exist",
        "route_contract_sha256": expected_route_digest,
        "results": results,
        "costs": costs,
    }


def _build_priority_sr_routes(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _build_priority_sr_route(
            route_id="projected_phase3_support_only_nph3_7",
            label="Projected Phase III: support only, no whitening",
            subtitle=(
                "50 rounds; n_ph=3 weak-Holstein / n_ph=7 strong-Holstein; "
                "accepted Powell refit remains supported-FS whitened"
            ),
            policy=(
                "full active-plus-singleton response; Gram-null support projection; "
                "direct generalized FS trust solve without Phase-III whitening; "
                "full accepted-ansatz supported-FS-whitened Powell refit"
            ),
            evidence=PROJECTED_PHASE3_EVIDENCE,
            expected_route_digest=PROJECTED_PHASE3_ROUTE_DIGEST,
            expected_trust_policy="displacement_calibrated_unbounded_v2",
            require_no_overlap=False,
            sources=sources,
        ),
        _build_priority_sr_route(
            route_id="no_overlap_trust_projected_phase3_nph3_7",
            label="No-overlap trust calibration",
            subtitle=(
                "50-round validated rows; n_ph=3 weak-Holstein / n_ph=7 "
                "strong-Holstein; all six regimes complete"
            ),
            policy=(
                "support-projected non-whitened Phase III; source-metric inverse-"
                "sqrt trust update with zero endpoint-overlap measurements; full "
                "accepted-ansatz supported-FS-whitened Powell refit"
            ),
            evidence=NO_OVERLAP_TRUST_EVIDENCE,
            expected_route_digest=NO_OVERLAP_TRUST_ROUTE_DIGEST,
            expected_trust_policy="source_metric_inverse_sqrt_no_overlap_v1",
            require_no_overlap=True,
            sources=sources,
        ),
    ]


def _upsert_priority_sr_routes(
    routes: Sequence[Mapping[str, Any]],
    *,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fresh = _build_priority_sr_routes(sources)
    fresh_ids = {str(route["id"]) for route in fresh}
    preserved = [
        json.loads(json.dumps(route))
        for route in routes
        if str(route.get("id")) not in fresh_ids
    ]
    insertion = next(
        (
            index + 1
            for index, route in enumerate(preserved)
            if str(route.get("id")) == "corrected_main_hysteresis_disabled_nph3_7"
        ),
        len(preserved),
    )
    return preserved[:insertion] + fresh + preserved[insertion:]


def _extract_comparator_trajectory(
    payload: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    result = payload.get("result")
    if payload.get("status") != "completed" or not isinstance(result, Mapping):
        raise RuntimeError("comparator result is not complete")
    history = result.get("adapt_history")
    if not isinstance(history, list) or not history:
        raise RuntimeError("comparator result has no ADAPT history")
    points: list[dict[str, Any]] = []
    for index, row in enumerate(history, start=1):
        if not isinstance(row, Mapping):
            continue
        round_id = _integer(row.get("outer_iteration"))
        if round_id is None:
            iteration = _integer(row.get("iteration"))
            round_id = iteration + 1 if iteration is not None else index
        error = _finite(row.get("abs_delta_e_same_cutoff_after"))
        if error is None:
            error = _finite(row.get("abs_delta_e_after"))
        if error is not None:
            points.append({"round": round_id, "error": error})
    if not points:
        raise RuntimeError("comparator history has no finite same-cutoff errors")
    terminal_error = _finite(receipt.get("same_cutoff_abs_error"))
    if terminal_error is None:
        terminal_error = _finite(result.get("abs_delta_e_same_cutoff"))
    if terminal_error is None:
        raise RuntimeError("comparator terminal same-cutoff error is missing")
    if points[-1]["round"] != 50:
        points.append({"round": 50, "error": terminal_error})
    return {
        "status": "complete",
        "n_ph": _integer(payload.get("n_ph_work")),
        "rounds": 50,
        "active_depth": _integer(receipt.get("active_depth")),
        "terminal_error": terminal_error,
        "s_alg": _integer(receipt.get("S_alg")),
        "s_alg_scope": "validated comparator ledger",
        "trajectory": points,
    }


def _comparator_tracking_summary(
    *,
    archive_path: Path,
    summary_path: Path,
    regime: str,
    expected_receipt_schema: str,
    expected_variant: str,
    expected_method_id: str,
) -> tuple[dict[str, Any], dict[str, int | None], list[dict[str, Any]]]:
    """Load a source-bound low-memory comparator summary and fail closed."""

    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if not summary_path.is_file():
        raise RuntimeError(
            f"compact tracking summary is required for late comparator archive: "
            f"{summary_path}"
        )
    summary = _read_json(summary_path)
    if summary.get("schema") != COMPARATOR_TRACKING_SUMMARY_SCHEMA:
        raise RuntimeError(f"comparator tracking-summary schema drift: {summary_path}")
    if summary.get("status") != "pass":
        raise RuntimeError(f"comparator tracking summary did not pass: {summary_path}")

    archive = summary.get("archive")
    if not isinstance(archive, Mapping):
        raise RuntimeError(f"tracking summary lacks archive identity: {summary_path}")
    declared_path = Path(str(archive.get("path") or ""))
    if not declared_path.is_absolute():
        declared_path = REPO_ROOT / declared_path
    if declared_path.resolve() != archive_path.resolve():
        raise RuntimeError(f"tracking-summary archive path drift: {summary_path}")
    observed_archive_sha = _sha256(archive_path)
    if str(archive.get("sha256") or "") != observed_archive_sha:
        raise RuntimeError(f"tracking-summary archive SHA-256 drift: {summary_path}")
    if _integer(archive.get("size_bytes")) != archive_path.stat().st_size:
        raise RuntimeError(f"tracking-summary archive size drift: {summary_path}")

    file_regime = COMPARATOR_REGIME_FILE[regime]
    job_id = f"{expected_variant}__{file_regime}__r50"
    identity = summary.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError(f"tracking summary lacks method identity: {summary_path}")
    expected_variant_slug = expected_variant.removeprefix("geo_").removeprefix(
        "append_"
    )
    if (
        identity.get("job_id") != job_id
        or identity.get("method_id") != expected_method_id
        or identity.get("variant") != expected_variant_slug
        or identity.get("same_cutoff_reference") is not True
    ):
        raise RuntimeError(f"tracking-summary route identity drift: {summary_path}")
    expected_n_ph = 3 if regime in REGIMES[:3] else 7
    if (
        _integer(identity.get("n_ph_work")) != expected_n_ph
        or _integer(identity.get("n_ph_reference")) != expected_n_ph
    ):
        raise RuntimeError(f"tracking-summary cutoff drift: {summary_path}")

    validation = summary.get("validation")
    if not isinstance(validation, Mapping):
        raise RuntimeError(f"tracking summary lacks validation authority: {summary_path}")
    if (
        validation.get("schema") != expected_receipt_schema
        or validation.get("status") != "pass"
        or validation.get("job_id") != job_id
        or validation.get("variant") != expected_variant_slug
        or _integer(validation.get("adapt_iterations")) != 50
        or validation.get("ledger_closure") != "pass"
        or validation.get("sector_leak_flag") is not False
        or validation.get("boson_truncation_leak_flag") is not False
    ):
        raise RuntimeError(f"tracking-summary validation gate drift: {summary_path}")

    result_member = summary.get("result_member")
    receipt_member = summary.get("validation_receipt_member")
    if not isinstance(result_member, Mapping) or not isinstance(receipt_member, Mapping):
        raise RuntimeError(f"tracking summary lacks member identities: {summary_path}")
    result_member_name = str(result_member.get("name") or "")
    receipt_member_name = str(receipt_member.get("name") or "")
    if not result_member_name.endswith(f"{job_id}/result.json"):
        raise RuntimeError(f"tracking-summary result member drift: {summary_path}")
    if not receipt_member_name.endswith(f"{job_id}/validation_receipt.json"):
        raise RuntimeError(f"tracking-summary receipt member drift: {summary_path}")
    for member, label in (
        (result_member, "result"),
        (receipt_member, "validation receipt"),
    ):
        if len(str(member.get("sha256") or "")) != 64:
            raise RuntimeError(f"tracking-summary {label} SHA-256 is absent: {summary_path}")
        if (_integer(member.get("size_bytes")) or 0) <= 0:
            raise RuntimeError(f"tracking-summary {label} size is invalid: {summary_path}")

    raw_result = summary.get("result")
    if not isinstance(raw_result, Mapping) or raw_result.get("status") != "complete":
        raise RuntimeError(f"tracking summary lacks a complete result: {summary_path}")
    trajectory = raw_result.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) != 50:
        raise RuntimeError(f"tracking-summary trajectory is not 50 rounds: {summary_path}")
    normalized_trajectory: list[dict[str, Any]] = []
    for expected_round, point in enumerate(trajectory, start=1):
        if not isinstance(point, Mapping):
            raise RuntimeError(f"tracking-summary trajectory row is malformed: {summary_path}")
        error = _finite(point.get("error"))
        if _integer(point.get("round")) != expected_round or error is None:
            raise RuntimeError(f"tracking-summary trajectory order/value drift: {summary_path}")
        normalized_trajectory.append({"round": expected_round, "error": abs(error)})
    terminal_error = _finite(raw_result.get("terminal_error"))
    if terminal_error is None or not math.isclose(
        terminal_error,
        float(validation.get("same_cutoff_abs_error")),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(f"tracking-summary terminal error drift: {summary_path}")
    active_depth = _integer(raw_result.get("active_depth"))
    s_alg = _integer(raw_result.get("s_alg"))
    if active_depth != _integer(validation.get("active_depth")):
        raise RuntimeError(f"tracking-summary active-depth drift: {summary_path}")
    if s_alg != _integer(validation.get("S_alg")):
        raise RuntimeError(f"tracking-summary S_alg drift: {summary_path}")

    qiskit = summary.get("qiskit")
    if not isinstance(qiskit, Mapping):
        raise RuntimeError(f"tracking summary lacks Qiskit costs: {summary_path}")
    costs = {
        metric: _integer(qiskit.get(metric)) for metric in ("N2q", "D2q", "Dc")
    }
    if any(costs[metric] is None for metric in ("N2q", "D2q", "Dc")):
        raise RuntimeError(f"tracking-summary Qiskit costs are incomplete: {summary_path}")

    archive_source = {
        "path": _relative(archive_path),
        "sha256": observed_archive_sha,
        "size_bytes": archive_path.stat().st_size,
        "member": result_member_name,
        "member_sha256": str(result_member["sha256"]),
    }
    receipt_source = dict(archive_source)
    receipt_source["member"] = receipt_member_name
    receipt_source["member_sha256"] = str(receipt_member["sha256"])
    summary_source = _source_record(summary_path)
    result = {
        "status": "complete",
        "n_ph": expected_n_ph,
        "rounds": 50,
        "active_depth": active_depth,
        "terminal_error": terminal_error,
        "s_alg": s_alg,
        "s_alg_scope": "validated comparator ledger",
        "trajectory": normalized_trajectory,
        "source": archive_source,
        "validation_receipt": receipt_source,
        "tracking_summary": summary_source,
    }
    return result, costs, [archive_source, receipt_source, summary_source]


def _build_comparator_archive_route(
    *,
    route_id: str,
    label: str,
    subtitle: str,
    policy: str,
    archives: Mapping[str, Path],
    expected_receipt_schema: str,
    expected_variant: str,
    expected_method_id: str,
    sources: list[dict[str, Any]],
    tracking_summaries: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    results = {regime: _empty_result("awaiting completed validation") for regime in REGIMES}
    costs = {regime: None for regime in REGIMES}
    for regime in REGIMES:
        archive_path = archives[regime]
        if not archive_path.is_file():
            continue
        summary_path = (
            tracking_summaries.get(regime)
            if tracking_summaries is not None
            else None
        )
        if summary_path is not None:
            result, cost, summary_sources = _comparator_tracking_summary(
                archive_path=archive_path,
                summary_path=summary_path,
                regime=regime,
                expected_receipt_schema=expected_receipt_schema,
                expected_variant=expected_variant,
                expected_method_id=expected_method_id,
            )
            results[regime] = result
            costs[regime] = cost
            sources.extend(summary_sources)
            continue
        file_regime = COMPARATOR_REGIME_FILE[regime]
        job_id = f"{expected_variant}__{file_regime}__r50"
        result_suffix = f"{job_id}/result.json"
        receipt_suffix = f"{job_id}/validation_receipt.json"
        members = _tar_json_members(archive_path, (result_suffix, receipt_suffix))
        payload, result_member = members[result_suffix]
        receipt, receipt_member = members[receipt_suffix]
        if receipt.get("schema") != expected_receipt_schema:
            raise RuntimeError(f"comparator receipt schema drift: {archive_path}")
        if receipt.get("status") != "pass" or receipt.get("job_id") != job_id:
            raise RuntimeError(f"comparator validation did not pass: {archive_path}")
        if receipt.get("variant") != expected_variant.removeprefix("geo_").removeprefix("append_"):
            raise RuntimeError(f"comparator variant drift: {archive_path}")
        if _integer(receipt.get("adapt_iterations")) != 50:
            raise RuntimeError(f"comparator did not complete 50 rounds: {archive_path}")
        if receipt.get("ledger_closure") != "pass":
            raise RuntimeError(f"comparator estimator ledger did not close: {archive_path}")
        if receipt.get("sector_leak_flag") is not False:
            raise RuntimeError(f"comparator sector leakage gate failed: {archive_path}")
        if receipt.get("boson_truncation_leak_flag") is not False:
            raise RuntimeError(f"comparator padding leakage gate failed: {archive_path}")
        if payload.get("method_id") != expected_method_id:
            raise RuntimeError(f"comparator method identity drift: {archive_path}")
        expected_n_ph = 3 if regime in REGIMES[:3] else 7
        if (
            _integer(payload.get("n_ph_work")) != expected_n_ph
            or _integer(payload.get("n_ph_reference")) != expected_n_ph
            or payload.get("same_cutoff_reference") is not True
        ):
            raise RuntimeError(f"comparator cutoff/reference drift: {archive_path}")

        result = _extract_comparator_trajectory(payload, receipt)
        archive_source = _source_record(archive_path, member=result_member)
        receipt_source = dict(archive_source)
        receipt_source["member"] = receipt_member
        result["source"] = archive_source
        result["validation_receipt"] = receipt_source
        results[regime] = result
        sources.extend((archive_source, receipt_source))
        costs[regime] = _qiskit_metrics(receipt)
        del payload, receipt, members
        gc.collect()

    return {
        "id": route_id,
        "label": label,
        "subtitle": subtitle,
        "policy": policy,
        "cost_convention": (
            "validated backend-free Table-I basis-gate compile, "
            "optimization level 0, seed 7"
        ),
        "results": results,
        "costs": costs,
    }


def _resolved_summary_artifact(
    record: Mapping[str, Any], *, summary_path: Path, label: str
) -> tuple[Path, dict[str, Any]]:
    """Resolve and verify one source record embedded in a compact summary."""

    path = Path(str(record.get("path") or ""))
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    sha = _sha256(path)
    if str(record.get("sha256") or "") != sha:
        raise RuntimeError(f"cost-arm {label} SHA-256 drift: {summary_path}")
    if _integer(record.get("size_bytes")) != path.stat().st_size:
        raise RuntimeError(f"cost-arm {label} size drift: {summary_path}")
    return path, {
        "path": _relative(path),
        "sha256": sha,
        "size_bytes": path.stat().st_size,
    }


def _cost_arm_tracking_summary(
    *,
    summary_path: Path,
    regime: str,
    expected_arm: str,
    expected_route_digest: str,
    expected_cost_mode: str,
    expected_fallback_policy: str,
) -> tuple[dict[str, Any], dict[str, int | None], list[dict[str, Any]]]:
    """Load one objectively passed cost-arm row; pending rows never enter."""

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = _read_json(summary_path)
    if (
        summary.get("schema") != COST_ARM_TRACKING_SUMMARY_SCHEMA
        or summary.get("status") != "pass"
    ):
        raise RuntimeError(f"cost-arm tracking summary is not a pass: {summary_path}")
    identity = summary.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError(f"cost-arm summary lacks route identity: {summary_path}")
    expected_n_ph = 3 if regime in REGIMES[:3] else 7
    if (
        identity.get("arm") != expected_arm
        or identity.get("regime") != regime
        or identity.get("profile_contract_sha256") != expected_route_digest
        or identity.get("cost_mode") != expected_cost_mode
        or identity.get("fallback_policy") != expected_fallback_policy
        or identity.get("same_cutoff_reference") is not True
        or _integer(identity.get("n_ph_work")) != expected_n_ph
        or _integer(identity.get("n_ph_reference")) != expected_n_ph
    ):
        raise RuntimeError(f"cost-arm tracking-summary identity drift: {summary_path}")

    archive_record = summary.get("archive")
    receipt_record = summary.get("revalidation_receipt")
    compact_record = summary.get("compact_trajectory_receipt")
    if not all(
        isinstance(record, Mapping)
        for record in (archive_record, receipt_record, compact_record)
    ):
        raise RuntimeError(f"cost-arm summary lacks source identities: {summary_path}")
    archive_path = Path(str(archive_record.get("path") or ""))
    if not archive_path.is_absolute():
        archive_path = REPO_ROOT / archive_path
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    archive_sha = str(archive_record.get("sha256") or "")
    if len(archive_sha) != 64:
        raise RuntimeError(f"cost-arm raw archive SHA-256 is absent: {summary_path}")
    if _integer(archive_record.get("size_bytes")) != archive_path.stat().st_size:
        raise RuntimeError(f"cost-arm raw archive size drift: {summary_path}")
    archive_source = {
        "path": _relative(archive_path),
        "sha256": archive_sha,
        "size_bytes": archive_path.stat().st_size,
        "sha256_validation_authority": (
            "v9 before/after preservation receipt plus compact streaming receipt"
        ),
        "raw_archive_not_reopened_by_tracker": True,
    }
    receipt_path, receipt_source = _resolved_summary_artifact(
        receipt_record, summary_path=summary_path, label="revalidation receipt"
    )
    _, compact_source = _resolved_summary_artifact(
        compact_record, summary_path=summary_path, label="compact trajectory receipt"
    )

    receipt = _read_json(receipt_path)
    if (
        receipt.get("schema") != COST_ARM_REVALIDATION_SCHEMAS[expected_arm]
        or receipt.get("status") != "pass"
        or receipt.get("scientific_rerun_required") is not False
        or receipt.get("raw_transfer_archive_preserved") is not True
        or receipt.get("regime_slug") != regime
        or receipt.get("profile_contract_sha256") != expected_route_digest
        or receipt.get("raw_transfer_archive_sha256_before")
        != archive_sha
        or receipt.get("raw_transfer_archive_sha256_after")
        != archive_sha
    ):
        raise RuntimeError(f"cost-arm revalidation receipt drift: {summary_path}")
    declared_archive = Path(str(receipt.get("raw_transfer_archive") or ""))
    if not declared_archive.is_absolute():
        declared_archive = receipt_path.parent / declared_archive
    if declared_archive.resolve() != archive_path.resolve():
        raise RuntimeError(f"cost-arm receipt/archive path drift: {summary_path}")

    result_member = summary.get("result_member")
    if not isinstance(result_member, Mapping):
        raise RuntimeError(f"cost-arm summary lacks result-member identity: {summary_path}")
    result_member_name = str(result_member.get("name") or "")
    result_member_sha = str(result_member.get("sha256") or "")
    if (
        not result_member_name.endswith(f"/{regime}/json/result.json")
        or len(result_member_sha) != 64
        or (_integer(result_member.get("size_bytes")) or 0) <= 0
    ):
        raise RuntimeError(f"cost-arm result-member identity drift: {summary_path}")

    executable_record = summary.get("executable_source")
    if not isinstance(executable_record, Mapping):
        raise RuntimeError(f"cost-arm summary lacks executable checkpoint: {summary_path}")
    executable_path, executable_source = _resolved_summary_artifact(
        executable_record,
        summary_path=summary_path,
        label="executable terminal checkpoint",
    )
    executable_source["trajectory_receipt_sha256"] = compact_source["sha256"]
    executable_source["trajectory_receipt_path"] = compact_source["path"]
    executable_checkpoint = _read_json(executable_path)
    checkpoint = executable_checkpoint.get("repaired_checkpoint")
    checkpoint_repair = executable_checkpoint.get("repair")
    checkpoint_source = executable_checkpoint.get("source")
    if not all(
        isinstance(record, Mapping)
        for record in (checkpoint, checkpoint_repair, checkpoint_source)
    ):
        raise RuntimeError(f"cost-arm executable checkpoint is malformed: {summary_path}")
    checkpoint_ledger = checkpoint.get("estimator_ledger_receipt")
    if not isinstance(checkpoint_ledger, Mapping):
        raise RuntimeError(f"cost-arm checkpoint lacks ledger identity: {summary_path}")

    validation = summary.get("validation")
    result_raw = summary.get("result")
    qiskit = summary.get("qiskit")
    prefix_qiskit = summary.get("terminal_prefix_qiskit")
    if not all(
        isinstance(record, Mapping)
        for record in (validation, result_raw, qiskit, prefix_qiskit)
    ):
        raise RuntimeError(f"cost-arm summary lacks validated result fields: {summary_path}")
    if (
        validation.get("status") != "pass"
        or _integer(validation.get("controller_rounds")) != 50
        or result_raw.get("status") != "complete"
        or _integer(result_raw.get("rounds")) != 50
        or _integer(result_raw.get("n_ph")) != expected_n_ph
    ):
        raise RuntimeError(f"cost-arm pass/horizon drift: {summary_path}")
    trajectory_raw = result_raw.get("trajectory")
    if not isinstance(trajectory_raw, list) or len(trajectory_raw) != 50:
        raise RuntimeError(f"cost-arm trajectory is not 50 rounds: {summary_path}")
    trajectory: list[dict[str, Any]] = []
    prior_s_alg = -1
    for expected_round, point in enumerate(trajectory_raw, start=1):
        if not isinstance(point, Mapping):
            raise RuntimeError(f"cost-arm trajectory row is malformed: {summary_path}")
        error = _finite(point.get("error"))
        active_depth = _integer(point.get("active_depth"))
        point_s_alg = _integer(point.get("S_alg"))
        if (
            _integer(point.get("round")) != expected_round
            or error is None
            or active_depth is None
            or active_depth < 0
            or point_s_alg is None
            or point_s_alg < prior_s_alg
            or not isinstance(point.get("prune_accepted"), bool)
        ):
            raise RuntimeError(f"cost-arm trajectory order/value drift: {summary_path}")
        trajectory.append(
            {
                "round": expected_round,
                "error": abs(error),
                "active_depth": active_depth,
                "prune_accepted": bool(point["prune_accepted"]),
                "S_alg": point_s_alg,
                "winning_lineage_S_alg": _integer(
                    point.get("winning_lineage_S_alg")
                ),
            }
        )
        prior_s_alg = point_s_alg
    trajectory_role = str(
        result_raw.get("trajectory_role") or "selected_terminal_path_v1"
    )
    selected_terminal_raw = result_raw.get("selected_terminal")
    if not isinstance(selected_terminal_raw, Mapping):
        raise RuntimeError(f"cost-arm selected-terminal receipt missing: {summary_path}")
    selected_round = _integer(selected_terminal_raw.get("round"))
    terminal_error = _finite(result_raw.get("terminal_error"))
    active_depth = _integer(result_raw.get("active_depth"))
    s_alg = _integer(result_raw.get("s_alg"))
    fidelity = _finite(result_raw.get("fidelity"))
    base_terminal_drift = (
        terminal_error is None
        or active_depth is None
        or s_alg is None
        or fidelity is None
        or trajectory[-1]["S_alg"] != s_alg
        or active_depth != _integer(validation.get("selected_final_active_depth"))
        or selected_round != _integer(validation.get("selected_final_controller_round"))
        or selected_round != _integer(selected_terminal_raw.get("round"))
        or active_depth != _integer(selected_terminal_raw.get("active_depth"))
        or not math.isclose(
            abs(_finite(selected_terminal_raw.get("error")) or math.inf),
            terminal_error or math.inf,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or s_alg != _integer(validation.get("all_branch_S_alg"))
        or s_alg
        != _integer(
            selected_terminal_raw.get("selection_authority_all_branch_S_alg")
        )
    )
    selected_winner_history: list[dict[str, Any]] | None = None
    controller_frontier: dict[str, Any] | None = None
    if trajectory_role == "selected_terminal_path_v1":
        terminal_drift = (
            base_terminal_drift
            or not math.isclose(
                trajectory[-1]["error"],
                terminal_error or math.inf,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or trajectory[-1]["active_depth"] != active_depth
            or selected_round != 50
        )
    elif trajectory_role == "controller_frontier_non_selected_v1":
        raw_history = result_raw.get("selected_winner_history")
        raw_frontier = result_raw.get("controller_frontier")
        if not isinstance(raw_history, list) or not isinstance(raw_frontier, Mapping):
            raise RuntimeError(
                f"cost-arm split terminal/frontier receipts missing: {summary_path}"
            )
        selected_winner_history = []
        for expected_round, point in enumerate(raw_history, start=1):
            if not isinstance(point, Mapping):
                raise RuntimeError(
                    f"cost-arm selected-winner history malformed: {summary_path}"
                )
            point_error = _finite(point.get("error"))
            point_depth = _integer(point.get("active_depth"))
            if (
                _integer(point.get("round")) != expected_round
                or point_error is None
                or point_depth is None
                or point_depth < 0
            ):
                raise RuntimeError(
                    f"cost-arm selected-winner history drift: {summary_path}"
                )
            selected_winner_history.append(
                {
                    "round": expected_round,
                    "error": abs(point_error),
                    "active_depth": point_depth,
                    "winning_lineage_S_alg": _integer(
                        point.get("winning_lineage_S_alg")
                    ),
                    "checkpoint_sha256": str(point.get("checkpoint_sha256") or ""),
                }
            )
        controller_frontier = dict(raw_frontier)
        terminal_drift = (
            base_terminal_drift
            or len(selected_winner_history) != selected_round
            or not selected_winner_history
            or not math.isclose(
                selected_winner_history[-1]["error"],
                terminal_error or math.inf,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or selected_winner_history[-1]["active_depth"] != active_depth
            or str(controller_frontier.get("status"))
            != "non_selected_recoverable_frontier"
            or _integer(controller_frontier.get("round")) != 50
            or _integer(controller_frontier.get("active_depth"))
            != trajectory[-1]["active_depth"]
            or not math.isclose(
                abs(_finite(controller_frontier.get("error")) or math.inf),
                trajectory[-1]["error"],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or controller_frontier.get(
                "eligible_for_selected_terminal_cost_reporting"
            )
            is not False
        )
    else:
        raise RuntimeError(
            f"unsupported cost-arm trajectory role {trajectory_role!r}: {summary_path}"
        )
    if terminal_drift:
        raise RuntimeError(f"cost-arm terminal/ledger drift: {summary_path}")
    if (
        executable_checkpoint.get("schema")
        != "paper_i_checkpoint_execution_order_repair_v1"
        or checkpoint_repair.get("status")
        not in {"repaired_permutation_only", "not_required"}
        or checkpoint_repair.get("substantive_term_changes") is not False
        or checkpoint_source.get("result_sha256") != result_member_sha
        or _integer(checkpoint_source.get("outer_iteration")) != selected_round
        or checkpoint.get("schema") != "paper_i_signed_active_prefix_checkpoint_v1"
        or _integer(checkpoint.get("outer_iteration")) != selected_round
        or _integer(checkpoint.get("active_ansatz_depth")) != active_depth
        or checkpoint.get("sr_route_profile_contract_sha256")
        != expected_route_digest
        or checkpoint_ledger.get("status") != "complete"
        or _integer(checkpoint_ledger.get("outer_iteration")) != selected_round
        or not isinstance(checkpoint.get("ordered_active_operator_labels"), list)
        or len(checkpoint["ordered_active_operator_labels"]) != active_depth
        or not isinstance(checkpoint.get("ordered_active_operators"), list)
        or len(checkpoint["ordered_active_operators"]) != active_depth
        or not isinstance(checkpoint.get("signed_unwrapped_logical_parameters"), list)
        or len(checkpoint["signed_unwrapped_logical_parameters"]) != active_depth
        or not isinstance(checkpoint.get("signed_unwrapped_runtime_parameters"), list)
        or not checkpoint["signed_unwrapped_runtime_parameters"]
    ):
        raise RuntimeError(
            f"cost-arm executable checkpoint/history/ledger drift: {summary_path}"
        )
    costs = {metric: _integer(qiskit.get(metric)) for metric in ("N2q", "D2q", "Dc")}
    if any(costs[metric] is None for metric in ("N2q", "D2q", "Dc")):
        raise RuntimeError(f"cost-arm Qiskit metrics are incomplete: {summary_path}")
    terminal_prefix_costs = {
        metric: _integer(prefix_qiskit.get(metric))
        for metric in ("N2q", "D2q", "Dc")
    }
    if any(
        terminal_prefix_costs[metric] is None for metric in ("N2q", "D2q", "Dc")
    ):
        raise RuntimeError(
            f"cost-arm terminal-prefix Qiskit metrics are incomplete: {summary_path}"
        )

    archive_source.update(
        {
            "member": result_member_name,
            "member_sha256": result_member_sha,
            "member_size_bytes": _integer(result_member.get("size_bytes")),
        }
    )
    summary_source = _source_record(summary_path)
    generated_sources: list[dict[str, Any]] = []
    generated = summary.get("generated_reporting_artifacts")
    if not isinstance(generated, Mapping):
        raise RuntimeError(f"cost-arm generated-artifact inventory missing: {summary_path}")
    for name, record in sorted(generated.items()):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"cost-arm generated source is malformed: {summary_path}")
        _, source = _resolved_summary_artifact(
            record, summary_path=summary_path, label=f"generated {name}"
        )
        generated_sources.append(source)
    sources = [
        archive_source,
        executable_source,
        receipt_source,
        compact_source,
        summary_source,
        *generated_sources,
    ]
    result = {
        "status": "complete",
        "n_ph": expected_n_ph,
        "rounds": 50,
        "active_depth": active_depth,
        "terminal_error": abs(terminal_error),
        "s_alg": s_alg,
        "s_alg_scope": str(result_raw.get("s_alg_scope") or ""),
        "fidelity": fidelity,
        "route_contract_sha256": expected_route_digest,
        "terminal_prefix_qiskit": terminal_prefix_costs,
        "trajectory_role": trajectory_role,
        "trajectory": trajectory,
        "selected_winner_history": selected_winner_history,
        "selected_terminal": dict(selected_terminal_raw),
        "controller_frontier": controller_frontier,
        "source": executable_source,
        "raw_archive_provenance": archive_source,
        "validation_receipt": receipt_source,
        "tracking_summary": summary_source,
    }
    return result, costs, sources


def _fmt_error(value: Any) -> str:
    parsed = _finite(value)
    return "--" if parsed is None else f"{parsed:.3e}"


def _fmt_int(value: Any) -> str:
    parsed = _integer(value)
    return "--" if parsed is None else f"{parsed:,}"


def _tex(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _attach_plateau_prefixes(
    routes: Sequence[dict[str, Any]],
    *,
    plateau_json: Path,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach fail-closed, source-matched plateau prefix receipts to routes."""

    if not plateau_json.is_file():
        raise FileNotFoundError(
            f"plateau-prefix sidecar is required before report construction: {plateau_json}"
        )
    payload = _read_json(plateau_json)
    if payload.get("schema") != PLATEAU_SCHEMA:
        raise RuntimeError(f"plateau sidecar schema drift: {payload.get('schema')!r}")
    rule = payload.get("rule")
    if not isinstance(rule, Mapping) or rule.get("id") != PLATEAU_RULE_ID:
        raise RuntimeError(f"plateau sidecar rule drift: {rule!r}")
    rows = payload.get("rows")
    unresolved = payload.get("unresolved")
    if not isinstance(rows, list) or not isinstance(unresolved, list):
        raise RuntimeError("plateau sidecar must contain rows and unresolved lists")

    row_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("plateau sidecar contains a non-object row")
        key = (str(row.get("route_id")), str(row.get("regime")))
        if key in row_map:
            raise RuntimeError(f"duplicate plateau row: {key}")
        if row.get("status") != "complete":
            raise RuntimeError(f"non-complete row appears in plateau rows: {key}")
        if row.get("rule", {}).get("id") != PLATEAU_RULE_ID:
            raise RuntimeError(f"plateau row rule drift: {key}")
        qiskit = row.get("qiskit")
        if not isinstance(qiskit, Mapping) or any(
            _integer(qiskit.get(metric)) is None for metric in ("N2q", "D2q", "Dc")
        ):
            raise RuntimeError(f"plateau row lacks compiled Qiskit metrics: {key}")
        if _integer(row.get("S_alg")) is None:
            raise RuntimeError(f"plateau row lacks prefix S_alg: {key}")
        row_map[key] = dict(row)

    unresolved_map = {
        (str(row.get("route_id")), str(row.get("regime"))): dict(row)
        for row in unresolved
        if isinstance(row, Mapping)
    }
    expected_complete = 0
    for route in routes:
        route_id = str(route["id"])
        route_plateau: dict[str, dict[str, Any]] = {}
        for regime in REGIMES:
            result = route["results"][regime]
            key = (route_id, regime)
            if result.get("trajectory"):
                row = row_map.get(key)
                if row is None:
                    unresolved_row = unresolved_map.get(key)
                    if route_id not in COST_ARM_ROUTE_IDS or unresolved_row is None:
                        raise RuntimeError(
                            f"completed result lacks plateau prefix receipt: {key}"
                        )
                    route_plateau[regime] = unresolved_row
                    continue
                expected_complete += 1
                result_source = result.get("source")
                row_source = row.get("source")
                if not isinstance(result_source, Mapping) or not isinstance(
                    row_source, Mapping
                ):
                    raise RuntimeError(f"plateau source receipt missing: {key}")
                if (
                    str(result_source.get("path")) != str(row_source.get("path"))
                    or str(result_source.get("sha256")) != str(row_source.get("sha256"))
                ):
                    raise RuntimeError(f"plateau source does not match displayed result: {key}")
                route_plateau[regime] = row
            else:
                row = unresolved_map.get(key)
                route_plateau[regime] = row or {
                    "route_id": route_id,
                    "regime": regime,
                    "status": "unresolved",
                    "reason": "no completed validated trajectory in tracker",
                }
        route["plateau"] = route_plateau
        route["plateau_cost_convention"] = (
            "backend-free Paper-I Table-I compile, optimization level 0, seed 7; "
            "exact reconstructed k_pl prefix including the reference state"
        )
    if expected_complete != len(row_map):
        extra = sorted(set(row_map) - {
            (str(route["id"]), regime)
            for route in routes
            for regime in REGIMES
            if route["results"][regime].get("trajectory")
        })
        raise RuntimeError(
            f"plateau row count mismatch: expected={expected_complete}, "
            f"observed={len(row_map)}, extra={extra}"
        )
    for row in row_map.values():
        prefix_source = row.get("prefix_source")
        if isinstance(prefix_source, Mapping):
            record = {
                "path": str(prefix_source.get("path")),
                "sha256": str(prefix_source.get("sha256")),
            }
            if prefix_source.get("result_member") is not None:
                record["member"] = str(prefix_source.get("result_member"))
            if prefix_source.get("result_member_sha256") is not None:
                record["member_sha256"] = str(
                    prefix_source.get("result_member_sha256")
                )
            sources.append(record)
    sources.append(
        {
            "path": _relative(REPO_ROOT / OUTPUT_RELATIVE_DIR / plateau_json.name),
            "sha256": _sha256(plateau_json),
            "size_bytes": plateau_json.stat().st_size,
        }
    )
    return dict(payload)


def _attach_target_prefixes(
    routes: Sequence[dict[str, Any]],
    *,
    target_json: Path,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach exact fixed-target first-crossing receipts to every route row."""

    if not target_json.is_file():
        raise FileNotFoundError(
            f"target-energy prefix sidecar is required before report construction: {target_json}"
        )
    payload = _read_json(target_json)
    if payload.get("schema") != TARGET_SCHEMA:
        raise RuntimeError(f"target sidecar schema drift: {payload.get('schema')!r}")
    rule = payload.get("rule")
    if (
        not isinstance(rule, Mapping)
        or rule.get("id") != TARGET_RULE_ID
        or not math.isclose(
            float(rule.get("target_abs_error", math.nan)),
            TARGET_ABS_ERROR,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise RuntimeError(f"target sidecar rule drift: {rule!r}")
    rows = payload.get("rows")
    unresolved = payload.get("unresolved")
    if not isinstance(rows, list) or not isinstance(unresolved, list):
        raise RuntimeError("target sidecar must contain rows and unresolved lists")

    row_map: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise RuntimeError("target sidecar contains a non-object row")
        row = dict(raw)
        key = (str(row.get("route_id")), str(row.get("regime")))
        if key in row_map:
            raise RuntimeError(f"duplicate target row: {key}")
        if row.get("status") != "complete":
            raise RuntimeError(f"non-complete row appears in target rows: {key}")
        qiskit = row.get("qiskit")
        if not isinstance(qiskit, Mapping) or any(
            _integer(qiskit.get(metric)) is None for metric in ("N2q", "D2q", "Dc")
        ):
            raise RuntimeError(f"target row lacks compiled Qiskit metrics: {key}")
        if _integer(row.get("S_alg")) is None:
            raise RuntimeError(f"target row lacks prefix S_alg: {key}")
        if float(row.get("error", math.inf)) > TARGET_ABS_ERROR:
            raise RuntimeError(f"target row does not reach the fixed threshold: {key}")
        row_map[key] = row

    unresolved_map = {
        (str(row.get("route_id")), str(row.get("regime"))): dict(row)
        for row in unresolved
        if isinstance(row, Mapping)
    }
    known_keys = {
        (str(route["id"]), regime)
        for route in routes
        for regime in REGIMES
    }
    extras = sorted((set(row_map) | set(unresolved_map)) - known_keys)
    if extras:
        raise RuntimeError(f"target sidecar contains unknown route rows: {extras}")

    for route in routes:
        route_id = str(route["id"])
        route_target: dict[str, dict[str, Any]] = {}
        for regime in REGIMES:
            result = route["results"][regime]
            key = (route_id, regime)
            row = row_map.get(key)
            unresolved_row = unresolved_map.get(key)
            if row is None and unresolved_row is None:
                raise RuntimeError(f"route row lacks target receipt or unresolved status: {key}")
            record = row if row is not None else unresolved_row
            assert record is not None
            if result.get("trajectory"):
                result_source = result.get("source")
                record_source = record.get("source")
                if not isinstance(result_source, Mapping) or not isinstance(
                    record_source, Mapping
                ):
                    raise RuntimeError(f"target source receipt missing: {key}")
                if (
                    str(result_source.get("path")) != str(record_source.get("path"))
                    or str(result_source.get("sha256")) != str(record_source.get("sha256"))
                ):
                    raise RuntimeError(f"target source does not match displayed result: {key}")
            route_target[regime] = dict(record)
        route["target_energy"] = route_target
        route["target_energy_cost_convention"] = (
            "first completed stored prefix with same-cutoff |Delta E| <= 2e-4; "
            "backend-free Paper-I Table-I compile, optimization level 0, seed 7, "
            "including the reference state"
        )

    for row in row_map.values():
        prefix_source = row.get("prefix_source")
        if isinstance(prefix_source, Mapping):
            record = {
                "path": str(prefix_source.get("path")),
                "sha256": str(prefix_source.get("sha256")),
            }
            if prefix_source.get("result_member") is not None:
                record["member"] = str(prefix_source.get("result_member"))
            if prefix_source.get("result_member_sha256") is not None:
                record["member_sha256"] = str(prefix_source.get("result_member_sha256"))
            sources.append(record)
    target_record = {
        "path": _relative(REPO_ROOT / OUTPUT_RELATIVE_DIR / target_json.name),
        "sha256": _sha256(target_json),
        "size_bytes": target_json.stat().st_size,
    }
    sources.append(target_record)
    return dict(payload)


def _plot_trajectory_grid(route: Mapping[str, Any], output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(7.35, 4.75), sharex=False)
    color = "#155D75"
    for axis, regime in zip(axes.flat, REGIMES, strict=True):
        result = route["results"][regime]
        trajectory = result.get("trajectory") or []
        if not trajectory:
            axis.set_facecolor("#F2F2F2")
            axis.text(
                0.5,
                0.55,
                "no completed\ntrajectory",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#666666",
                fontsize=9,
            )
            axis.text(
                0.5,
                0.24,
                str(result.get("status") or "pending"),
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#888888",
                fontsize=7.5,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        else:
            rounds = [int(point["round"]) for point in trajectory]
            errors = [max(float(point["error"]), 1e-16) for point in trajectory]
            frontier_only = (
                result.get("trajectory_role")
                == "controller_frontier_non_selected_v1"
            )
            axis.semilogy(
                rounds,
                errors,
                color=color,
                marker="o",
                ms=2.0,
                lw=1.25,
                ls="--" if frontier_only else "-",
                alpha=0.88 if frontier_only else 1.0,
            )
            if frontier_only:
                selected_terminal = result.get("selected_terminal")
                if not isinstance(selected_terminal, Mapping):
                    raise RuntimeError(
                        f"frontier-only route lacks selected terminal: {route['id']}/{regime}"
                    )
                axis.scatter(
                    [int(selected_terminal["round"])],
                    [max(float(selected_terminal["error"]), 1e-16)],
                    marker="D",
                    s=32,
                    color="#B23A48",
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=6,
                )
            plateau = route.get("plateau", {}).get(regime)
            if isinstance(plateau, Mapping) and plateau.get("status") == "complete":
                k_pl = int(plateau["k_pl"])
                plateau_error = max(float(plateau["error"]), 1e-16)
                axis.axvline(k_pl, color="#B23A48", lw=0.8, ls="--", alpha=0.8)
                axis.scatter(
                    [k_pl],
                    [plateau_error],
                    marker="*",
                    s=38,
                    color="#B23A48",
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=5,
                )
            axis.grid(True, which="both", alpha=0.22)
            axis.set_xlabel("round", fontsize=7.8)
            axis.tick_params(labelsize=7.2)
            nph = result.get("n_ph")
            if frontier_only:
                terminal_label = (
                    f"selected k={int(result['selected_terminal']['round'])}"
                    f"\n|dE|={_fmt_error(result.get('terminal_error'))}"
                )
                frontier_label = "\nfrontier k=50 (not selected)"
            else:
                terminal_label = f"final={_fmt_error(result.get('terminal_error'))}"
                frontier_label = ""
            axis.text(
                0.03,
                0.05,
                (
                    f"n_ph={nph}; {terminal_label}"
                    + frontier_label
                    + (
                        f"\nk_pl={int(plateau['k_pl'])}"
                        if isinstance(plateau, Mapping)
                        and plateau.get("status") == "complete"
                        else ""
                    )
                ),
                transform=axis.transAxes,
                fontsize=6.8,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
        axis.set_title(REGIME_SHORT[regime], fontsize=9.5, fontweight="bold")
    axes[0, 0].set_ylabel("same-cutoff |E-E_ED|", fontsize=8)
    axes[1, 0].set_ylabel("same-cutoff |E-E_ED|", fontsize=8)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    path = output_dir / f"trajectory_{route['id']}.png"
    fig.savefig(path, dpi=260)
    plt.close(fig)
    return path


def _build_top_sr_append_plateau_comparison(
    routes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve the matched corrected SR pair and projected-singleton comparator."""

    route_map = {str(route["id"]): route for route in routes}
    required_ids = (*TOP_SR_ROUTE_IDS, APPEND_PROJECTED_ROUTE_ID)
    missing = [route_id for route_id in required_ids if route_id not in route_map]
    if missing:
        raise RuntimeError(f"comparison routes missing from tracker: {missing}")

    ranked: list[dict[str, Any]] = []
    for route_id in TOP_SR_ROUTE_IDS:
        route = route_map[route_id]
        plateau_errors: list[float] = []
        for index, regime in enumerate(REGIMES):
            result = route["results"][regime]
            plateau = route.get("plateau", {}).get(regime)
            if not result.get("trajectory"):
                raise RuntimeError(
                    f"top-SR comparison route lacks a completed trajectory: "
                    f"{route_id}/{regime}"
                )
            if not isinstance(plateau, Mapping) or plateau.get("status") != "complete":
                raise RuntimeError(
                    f"top-SR comparison route lacks an exact plateau receipt: "
                    f"{route_id}/{regime}"
                )
            expected_nph = 3 if index < 3 else 7
            if _integer(result.get("n_ph")) != expected_nph:
                raise RuntimeError(
                    f"top-SR comparison cutoff drift: {route_id}/{regime}"
                )
            error = float(plateau["error"])
            if not math.isfinite(error) or error <= 0.0:
                raise RuntimeError(
                    f"top-SR comparison has invalid plateau error: {route_id}/{regime}"
                )
            plateau_errors.append(error)
        ranked.append(
            {
                "route_id": route_id,
                "label": str(route["label"]),
                "geometric_mean_plateau_error": math.exp(
                    sum(math.log(error) for error in plateau_errors)
                    / len(plateau_errors)
                ),
            }
        )
    ranked.sort(key=lambda row: float(row["geometric_mean_plateau_error"]))

    append_route = route_map[APPEND_PROJECTED_ROUTE_ID]
    common_regimes: list[str] = []
    unresolved_regimes: list[str] = []
    for regime in REGIMES:
        append_result = append_route["results"][regime]
        append_plateau = append_route.get("plateau", {}).get(regime)
        if (
            append_result.get("trajectory")
            and isinstance(append_plateau, Mapping)
            and append_plateau.get("status") == "complete"
        ):
            expected_nph = 3 if REGIMES.index(regime) < 3 else 7
            if _integer(append_result.get("n_ph")) != expected_nph:
                raise RuntimeError(
                    f"Append projected-singleton comparison cutoff drift: {regime}"
                )
            common_regimes.append(regime)
        else:
            unresolved_regimes.append(regime)
    if not common_regimes:
        raise RuntimeError(
            "Append projected-singleton has no exact plateau rows for comparison"
        )

    row_route_ids = [row["route_id"] for row in ranked] + [APPEND_PROJECTED_ROUTE_ID]
    rows: dict[str, dict[str, Any]] = {}
    for route_id in row_route_ids:
        route = route_map[route_id]
        route_rows: dict[str, Any] = {}
        for regime in common_regimes:
            plateau = route["plateau"][regime]
            qiskit = plateau["qiskit"]
            route_rows[regime] = {
                "n_ph": _integer(route["results"][regime].get("n_ph")),
                "k_pl": _integer(plateau.get("k_pl")),
                "error": float(plateau["error"]),
                "S_alg": _integer(plateau.get("S_alg")),
                "qiskit": {
                    metric: _integer(qiskit.get(metric))
                    for metric in ("N2q", "D2q", "Dc")
                },
            }
        rows[route_id] = route_rows

    return {
        "schema": COMPARISON_SCHEMA,
        "selection_policy": COMPARISON_SELECTION_POLICY,
        "ranking_metric": (
            "geometric mean of exact plateau same-cutoff errors over all six "
            "completed n_ph=3/7 regimes"
        ),
        "snake_routes": ranked,
        "append_route": {
            "route_id": APPEND_PROJECTED_ROUTE_ID,
            "label": str(append_route["label"]),
        },
        "route_ids": row_route_ids,
        "common_regimes": common_regimes,
        "unresolved_append_regimes": unresolved_regimes,
        "marker_policy": "one method-specific marker per curve at exact k_pl",
        "trajectory_point_policy": (
            "use every stored accepted-history point; no synthetic round-zero point"
        ),
        "rows": rows,
    }


def _build_top_sr_append_target_comparison(
    routes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    route_map = {str(route["id"]): route for route in routes}
    route_ids = [*TOP_SR_ROUTE_IDS, APPEND_PROJECTED_ROUTE_ID]
    regimes = list(REGIMES[:3])
    rows: dict[str, dict[str, Any]] = {}
    for route_id in route_ids:
        route = route_map.get(route_id)
        if route is None:
            raise RuntimeError(f"target comparison route missing: {route_id}")
        route_rows: dict[str, Any] = {}
        for regime in regimes:
            result = route["results"][regime]
            record = route.get("target_energy", {}).get(regime)
            if _integer(result.get("n_ph")) != 3:
                raise RuntimeError(f"target comparison cutoff drift: {route_id}/{regime}")
            if not isinstance(record, Mapping) or record.get("status") != "complete":
                raise RuntimeError(f"target comparison prefix unavailable: {route_id}/{regime}")
            qiskit = record.get("qiskit")
            if not isinstance(qiskit, Mapping):
                raise RuntimeError(f"target comparison Qiskit receipt unavailable: {route_id}/{regime}")
            route_rows[regime] = {
                "n_ph": 3,
                "k_target": _integer(record.get("k_target")),
                "active_depth": _integer(record.get("active_depth")),
                "error": float(record["error"]),
                "S_alg": _integer(record.get("S_alg")),
                "qiskit": {
                    metric: _integer(qiskit.get(metric))
                    for metric in ("N2q", "D2q", "Dc")
                },
            }
        rows[route_id] = route_rows
    return {
        "schema": TARGET_COMPARISON_SCHEMA,
        "selection_policy": TARGET_RULE_ID,
        "target_abs_error": TARGET_ABS_ERROR,
        "physical_target_definition": "E_T = 1e-4 L E0 = 2e-4",
        "route_ids": route_ids,
        "common_regimes": regimes,
        "marker_policy": "one method-specific marker at the exact first target crossing",
        "trajectory_point_policy": "use every stored accepted-history point; no interpolation",
        "rows": rows,
    }


def _build_method_representation_comparison(
    routes: Sequence[Mapping[str, Any]],
    *,
    representation: str,
) -> dict[str, Any]:
    """Compare SNAKE, Geo-ADAPT, and Append-ADAPT at one pool resolution."""

    route_ids = METHOD_REPRESENTATION_ROUTE_IDS.get(representation)
    if route_ids is None:
        raise ValueError(f"unsupported candidate representation: {representation}")
    route_map = {str(route["id"]): route for route in routes}
    missing = [route_id for route_id in route_ids if route_id not in route_map]
    if missing:
        raise RuntimeError(
            f"{representation} comparison routes missing from tracker: {missing}"
        )

    rows: dict[str, dict[str, Any]] = {}
    hit_counts: dict[str, int] = {}
    for route_id in route_ids:
        route = route_map[route_id]
        route_rows: dict[str, Any] = {}
        hit_count = 0
        for index, regime in enumerate(REGIMES):
            result = route["results"][regime]
            trajectory = result.get("trajectory")
            if not isinstance(trajectory, list) or not trajectory:
                raise RuntimeError(
                    f"{representation} comparison lacks trajectory: "
                    f"{route_id}/{regime}"
                )
            expected_nph = 3 if index < 3 else 7
            if _integer(result.get("n_ph")) != expected_nph:
                raise RuntimeError(
                    f"{representation} comparison cutoff drift: "
                    f"{route_id}/{regime}"
                )

            target = route.get("target_energy", {}).get(regime)
            target = target if isinstance(target, Mapping) else {}
            target_hit = target.get("status") == "complete"
            if target_hit:
                qiskit = target.get("qiskit")
                if not isinstance(qiskit, Mapping):
                    raise RuntimeError(
                        f"{representation} target Qiskit receipt unavailable: "
                        f"{route_id}/{regime}"
                    )
                selected_round = _integer(target.get("k_target"))
                selected_error = float(target["error"])
                selected_s_alg = _integer(target.get("S_alg"))
                hit_count += 1
            else:
                qiskit = route.get("costs", {}).get(regime)
                if not isinstance(qiskit, Mapping):
                    raise RuntimeError(
                        f"{representation} terminal Qiskit receipt unavailable: "
                        f"{route_id}/{regime}"
                    )
                selected_round = _integer(result.get("rounds"))
                selected_error = float(result["terminal_error"])
                selected_s_alg = _integer(result.get("s_alg"))

            if selected_round is None or selected_s_alg is None:
                raise RuntimeError(
                    f"{representation} selected endpoint incomplete: "
                    f"{route_id}/{regime}"
                )
            if not math.isfinite(selected_error) or selected_error < 0.0:
                raise RuntimeError(
                    f"{representation} selected error invalid: "
                    f"{route_id}/{regime}"
                )
            route_rows[regime] = {
                "n_ph": expected_nph,
                "endpoint": "target_crossing" if target_hit else "terminal_nonhit",
                "target_hit": target_hit,
                "round": selected_round,
                "error": selected_error,
                "S_alg": selected_s_alg,
                "qiskit": {
                    metric: _integer(qiskit.get(metric))
                    for metric in ("N2q", "D2q", "Dc")
                },
            }
        rows[route_id] = route_rows
        hit_counts[route_id] = hit_count

    return {
        "schema": METHOD_REPRESENTATION_COMPARISON_SCHEMA,
        "representation": representation,
        "route_ids": list(route_ids),
        "regimes": list(REGIMES),
        "target_abs_error": TARGET_ABS_ERROR,
        "physical_target_definition": "E_T = 1e-4 L E0 = 2e-4",
        "endpoint_policy": (
            "first exact target crossing when available; otherwise the "
            "validated terminal k=50 record"
        ),
        "marker_policy": (
            "one method-specific marker per curve; filled at the first target "
            "crossing and hollow at a terminal non-hit"
        ),
        "hit_counts": hit_counts,
        "rows": rows,
    }


def _plot_method_representation_comparison(
    routes: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    route_map = {str(route["id"]): route for route in routes}
    styles = {
        "sr_macro_physical_lanes_nph3_7": {
            "label": "SNAKE macro",
            "color": "#E45756",
            "marker": "*",
            "line_width": 1.7,
            "marker_size": 58,
        },
        "sr_guarded_singleton_no_lanes_nph3_7": {
            "label": "SNAKE singleton",
            "color": "#E45756",
            "marker": "*",
            "line_width": 1.7,
            "marker_size": 58,
        },
        "geo_adapt_macro_nph3_7": {
            "label": "Geo-ADAPT macro",
            "color": "#54A24B",
            "marker": "^",
            "line_width": 1.4,
            "marker_size": 30,
        },
        "geo_adapt_projected_singleton_nph3_7": {
            "label": "Geo-ADAPT singleton",
            "color": "#54A24B",
            "marker": "^",
            "line_width": 1.4,
            "marker_size": 30,
        },
        "append_adapt_macro_nph3_7": {
            "label": "Append-ADAPT macro",
            "color": "#4C78A8",
            "marker": "o",
            "line_width": 1.4,
            "marker_size": 30,
        },
        "append_adapt_projected_singleton_nph3_7": {
            "label": "Append-ADAPT singleton",
            "color": "#4C78A8",
            "marker": "o",
            "line_width": 1.4,
            "marker_size": 30,
        },
    }
    route_ids = list(comparison["route_ids"])
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 5.25), squeeze=False)
    for axis, regime in zip(axes.flat, REGIMES, strict=True):
        max_round = 0
        for route_id in route_ids:
            route = route_map[route_id]
            trajectory = route["results"][regime]["trajectory"]
            rounds = [int(point["round"]) for point in trajectory]
            errors = [max(float(point["error"]), 1e-16) for point in trajectory]
            max_round = max(max_round, max(rounds))
            style = styles[route_id]
            axis.plot(
                rounds,
                errors,
                color=style["color"],
                linestyle="-",
                linewidth=style["line_width"],
                alpha=0.96,
            )
            selected = comparison["rows"][route_id][regime]
            axis.scatter(
                [int(selected["round"])],
                [max(float(selected["error"]), 1e-16)],
                color=style["color"],
                facecolor=style["color"] if selected["target_hit"] else "white",
                marker=style["marker"],
                s=style["marker_size"],
                edgecolor=style["color"],
                linewidth=0.9,
                zorder=5,
            )
        axis.set_yscale("log")
        axis.set_xlim(0, max_round)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        axis.set_xlabel("ADAPT selection round", fontsize=7.6)
        axis.set_title(
            f"{REGIME_SHORT[regime]}  "
            f"(n_ph={comparison['rows'][route_ids[0]][regime]['n_ph']})",
            fontsize=9.2,
            fontweight="bold",
        )
        axis.tick_params(labelsize=6.9)
        axis.grid(True, which="both", alpha=0.22, linewidth=0.45)
    axes[0, 0].set_ylabel(r"same-cutoff $|E-E_{\rm ED}|$", fontsize=7.6)
    axes[1, 0].set_ylabel(r"same-cutoff $|E-E_{\rm ED}|$", fontsize=7.6)

    handles = []
    for route_id in route_ids:
        style = styles[route_id]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle="-",
                linewidth=style["line_width"],
                marker=style["marker"],
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                markersize=6.5 if style["marker"] == "*" else 4.5,
                label=style["label"],
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.2,
        title=(
            r"filled marker = first $|\Delta E|\leq E_T$; "
            r"hollow marker = terminal non-hit"
        ),
        title_fontsize=6.8,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0), h_pad=0.8, w_pad=0.8)
    representation = str(comparison["representation"])
    path = output_dir / (
        f"comparison_three_method_{representation}_target_or_terminal.png"
    )
    fig.savefig(path, dpi=280)
    plt.close(fig)
    return path


def _plot_top_sr_append_plateau_comparison(
    routes: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any],
    output_dir: Path,
    *,
    scope: str = "plateau",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    route_map = {str(route["id"]): route for route in routes}
    if scope not in {"plateau", "target_energy"}:
        raise ValueError(f"unsupported comparison marker scope: {scope}")
    record_key = "plateau" if scope == "plateau" else "target_energy"
    marker_key = "k_pl" if scope == "plateau" else "k_target"
    regimes = list(comparison["common_regimes"])
    ncols = min(3, len(regimes))
    nrows = int(math.ceil(len(regimes) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.35, 2.42 * nrows + 0.38),
        squeeze=False,
    )
    styles = {
        "corrected_main_hysteresis_disabled_nph3_7": {
            "label": "SR main",
            "color": "#E45756",
            "marker": "*",
            "marker_face": "#E45756",
            "line_width": 1.7,
            "marker_size": 58,
        },
        "corrected_fs_prune_hysteresis_disabled_nph3_7": {
            "label": "SR FS-prune",
            "color": "#9C2F3F",
            "marker": "*",
            "marker_face": "white",
            "line_width": 1.45,
            "marker_size": 58,
        },
        APPEND_PROJECTED_ROUTE_ID: {
            "label": "Append projected-singleton",
            "color": "#4C78A8",
            "marker": "o",
            "marker_face": "#4C78A8",
            "line_width": 1.4,
            "marker_size": 30,
        },
    }
    route_ids = list(comparison["route_ids"])
    for axis, regime in zip(axes.flat, regimes, strict=False):
        max_round = 0
        for route_id in route_ids:
            route = route_map[route_id]
            trajectory = route["results"][regime]["trajectory"]
            rounds = [int(point["round"]) for point in trajectory]
            errors = [max(float(point["error"]), 1e-16) for point in trajectory]
            max_round = max(max_round, max(rounds))
            style = styles[route_id]
            axis.plot(
                rounds,
                errors,
                color=style["color"],
                linestyle="-",
                linewidth=style["line_width"],
                alpha=0.96,
            )
            selected = route[record_key][regime]
            axis.scatter(
                [int(selected[marker_key])],
                [max(float(selected["error"]), 1e-16)],
                color=style["color"],
                facecolor=style["marker_face"],
                marker=style["marker"],
                s=style["marker_size"],
                edgecolor=style["color"],
                linewidth=0.8,
                zorder=5,
            )
        axis.set_yscale("log")
        axis.set_xlim(0, max_round)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        axis.set_xlabel("ADAPT selection round", fontsize=7.6)
        axis.set_title(
            f"{REGIME_SHORT[regime]}  (n_ph={route_map[APPEND_PROJECTED_ROUTE_ID]['results'][regime]['n_ph']})",
            fontsize=9.2,
            fontweight="bold",
        )
        axis.tick_params(labelsize=6.9)
        axis.grid(True, which="both", alpha=0.22, linewidth=0.45)
    for index, axis in enumerate(axes.flat):
        if index >= len(regimes):
            axis.axis("off")
        elif index % ncols == 0:
            axis.set_ylabel(r"same-cutoff $|E-E_{\rm ED}|$", fontsize=7.6)

    handles = []
    for route_id in route_ids:
        style = styles[route_id]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle="-",
                linewidth=style["line_width"],
                marker=style["marker"],
                markerfacecolor=style["marker_face"],
                markeredgecolor=style["color"],
                markersize=6.5 if style["marker"] == "*" else 4.5,
                label=style["label"],
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.2,
        title=(
            r"single marker = exact $k_{\rm pl}$"
            if scope == "plateau"
            else r"single marker = first $k_T$ with $|\Delta E|\leq E_T$"
        ),
        title_fontsize=6.8,
    )
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0), h_pad=0.8, w_pad=0.8)
    path = output_dir / (
        "comparison_top2_sr_vs_append_projected_plateau.png"
        if scope == "plateau"
        else "comparison_top2_sr_vs_append_projected_target_energy.png"
    )
    fig.savefig(path, dpi=280)
    plt.close(fig)
    return path


def _plot_cost_grid(
    route: Mapping[str, Any], output_dir: Path, *, scope: str = "endpoint"
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 4, figsize=(7.35, 2.2))
    x = np.arange(len(REGIMES), dtype=float)
    labels = [REGIME_SHORT[regime] for regime in REGIMES]
    colors = {
        "S_alg": "#3A506B",
        "N2q": "#186F8C",
        "D2q": "#D17832",
        "Dc": "#6C5B7B",
    }
    for axis, metric in zip(axes, ("S_alg", "N2q", "D2q", "Dc"), strict=True):
        any_value = False
        for index, regime in enumerate(REGIMES):
            if scope == "plateau":
                plateau = route.get("plateau", {}).get(regime)
                record = plateau if isinstance(plateau, Mapping) else {}
                if metric == "S_alg":
                    value = _integer(record.get("S_alg"))
                else:
                    qiskit = record.get("qiskit")
                    value = (
                        _integer(qiskit.get(metric))
                        if isinstance(qiskit, Mapping)
                        else None
                    )
            elif scope == "target_energy":
                target = route.get("target_energy", {}).get(regime)
                record = target if isinstance(target, Mapping) else {}
                if metric == "S_alg":
                    value = _integer(record.get("S_alg"))
                else:
                    qiskit = record.get("qiskit")
                    value = (
                        _integer(qiskit.get(metric))
                        if isinstance(qiskit, Mapping)
                        else None
                    )
            elif metric == "S_alg":
                value = _integer(route["results"][regime].get("s_alg"))
            else:
                record = route["costs"].get(regime)
                value = _integer(record.get(metric)) if isinstance(record, Mapping) else None
            if value is None:
                axis.text(index, 0.02, "--", ha="center", va="bottom", color="#888888", fontsize=8)
                continue
            any_value = True
            axis.bar(index, value, color=colors[metric], width=0.68)
            axis.text(index, value, f"{value}", ha="center", va="bottom", fontsize=6.5, rotation=90)
        axis.set_xticks(x, labels)
        axis.tick_params(labelsize=7.2)
        title = r"$S_{\mathrm{alg}}$" if metric == "S_alg" else metric
        axis.set_title(title, fontsize=9.5, fontweight="bold")
        axis.grid(True, axis="y", alpha=0.2)
        if not any_value:
            axis.set_ylim(0, 1)
            axis.set_yticks([])
            axis.text(0.5, 0.52, "not compiled", transform=axis.transAxes, ha="center", color="#777777")
    fig.tight_layout(w_pad=0.8)
    suffix = "" if scope == "endpoint" else f"_{scope}"
    path = output_dir / f"qiskit{suffix}_{route['id']}.png"
    fig.savefig(path, dpi=260)
    plt.close(fig)
    return path


def _cost_table(route: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Regime & $n_{\rm ph}$ & $|\Delta E|$ & $S_{\mathrm{alg}}$ & $N_{2q}$ & $D_{2q}$ & $D_c$ \\",
        r"\midrule",
    ]
    for regime in REGIMES:
        record = route["costs"].get(regime)
        record = record if isinstance(record, Mapping) else {}
        lines.append(
            f"{REGIME_TEX[regime]} & {_fmt_int(route['results'][regime].get('n_ph'))} & "
            f"{_fmt_error(route['results'][regime].get('terminal_error'))} & "
            f"{_fmt_int(route['results'][regime].get('s_alg'))} & "
            f"{_fmt_int(record.get('N2q'))} & "
            f"{_fmt_int(record.get('D2q'))} & {_fmt_int(record.get('Dc'))} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _target_energy_cost_table(route: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Regime & $n_{\rm ph}$ & $k_T$ & $|\Delta E_{k_T}|$ & $S_{\mathrm{alg}}$ & $N_{2q}$ & $D_{2q}$ & $D_c$ \\",
        r"\midrule",
    ]
    for regime in REGIMES:
        record = route.get("target_energy", {}).get(regime)
        record = record if isinstance(record, Mapping) else {}
        qiskit = record.get("qiskit")
        qiskit = qiskit if isinstance(qiskit, Mapping) else {}
        if record.get("status") == "complete":
            cells = (
                _fmt_int(record.get("k_target")),
                _fmt_error(record.get("error")),
                _fmt_int(record.get("S_alg")),
                _fmt_int(qiskit.get("N2q")),
                _fmt_int(qiskit.get("D2q")),
                _fmt_int(qiskit.get("Dc")),
            )
            lines.append(
                f"{REGIME_TEX[regime]} & {_fmt_int(route['results'][regime].get('n_ph'))} & "
                + " & ".join(cells)
                + r" \\"
            )
        else:
            lines.append(
                f"{REGIME_TEX[regime]} & {_fmt_int(route['results'][regime].get('n_ph'))} & "
                + r"\multicolumn{6}{c}{not reached; best $|\Delta E|="
                + _fmt_error(record.get("best_observed_error"))
                + r"$} \\"
            )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _plateau_cost_table(route: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Regime & $n_{\rm ph}$ & $k_{\rm pl}$ & $|\Delta E|$ & $S_{\mathrm{alg}}$ & $N_{2q}$ & $D_{2q}$ & $D_c$ \\",
        r"\midrule",
    ]
    for regime in REGIMES:
        record = route.get("plateau", {}).get(regime)
        record = record if isinstance(record, Mapping) else {}
        qiskit = record.get("qiskit")
        qiskit = qiskit if isinstance(qiskit, Mapping) else {}
        lines.append(
            f"{REGIME_TEX[regime]} & {_fmt_int(route['results'][regime].get('n_ph'))} & "
            f"{_fmt_int(record.get('k_pl'))} & {_fmt_error(record.get('error'))} & "
            f"{_fmt_int(record.get('S_alg'))} & {_fmt_int(qiskit.get('N2q'))} & "
            f"{_fmt_int(qiskit.get('D2q'))} & {_fmt_int(qiskit.get('Dc'))} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _comparison_plateau_cost_table(
    route: Mapping[str, Any], regimes: Sequence[str]
) -> str:
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Reg. & $k_{\rm pl}$ & $|\Delta E|$ & $S_{\rm alg}$ & $N_{2q}$ & $D_{2q}$ & $D_c$ \\",
        r"\midrule",
    ]
    for regime in regimes:
        record = route.get("plateau", {}).get(regime)
        record = record if isinstance(record, Mapping) else {}
        qiskit = record.get("qiskit")
        qiskit = qiskit if isinstance(qiskit, Mapping) else {}
        lines.append(
            f"{REGIME_SHORT[regime]} & {_fmt_int(record.get('k_pl'))} & "
            f"{_fmt_error(record.get('error'))} & {_fmt_int(record.get('S_alg'))} & "
            f"{_fmt_int(qiskit.get('N2q'))} & {_fmt_int(qiskit.get('D2q'))} & "
            f"{_fmt_int(qiskit.get('Dc'))} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _comparison_target_cost_table(
    route: Mapping[str, Any], regimes: Sequence[str]
) -> str:
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Reg. & $k_T$ & $|\Delta E|$ & $S_{\rm alg}$ & $N_{2q}$ & $D_{2q}$ & $D_c$ \\",
        r"\midrule",
    ]
    for regime in regimes:
        record = route.get("target_energy", {}).get(regime)
        record = record if isinstance(record, Mapping) else {}
        qiskit = record.get("qiskit")
        qiskit = qiskit if isinstance(qiskit, Mapping) else {}
        lines.append(
            f"{REGIME_SHORT[regime]} & {_fmt_int(record.get('k_target'))} & "
            f"{_fmt_error(record.get('error'))} & {_fmt_int(record.get('S_alg'))} & "
            f"{_fmt_int(qiskit.get('N2q'))} & {_fmt_int(qiskit.get('D2q'))} & "
            f"{_fmt_int(qiskit.get('Dc'))} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _method_representation_cost_table(
    comparison: Mapping[str, Any],
    route_id: str,
) -> str:
    lines = [
        r"\begin{tabular}{@{}lcrrrrrr@{}}",
        r"\toprule",
        r"Reg. & Hit & $k$ & $|\Delta E|$ & $S_{\rm alg}$ & $N_{2q}$ & $D_{2q}$ & $D_c$ \\",
        r"\midrule",
    ]
    rows = comparison["rows"][route_id]
    for regime in REGIMES:
        record = rows[regime]
        qiskit = record["qiskit"]
        lines.append(
            f"{REGIME_SHORT[regime]} & "
            f"{'yes' if record['target_hit'] else 'no'} & "
            f"{_fmt_int(record.get('round'))} & "
            f"{_fmt_error(record.get('error'))} & "
            f"{_fmt_int(record.get('S_alg'))} & "
            f"{_fmt_int(qiskit.get('N2q'))} & "
            f"{_fmt_int(qiskit.get('D2q'))} & "
            f"{_fmt_int(qiskit.get('Dc'))} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _status_table(routes: Sequence[Mapping[str, Any]]) -> str:
    compact_labels = {
        "sr_macro_beam3x2_fs_prune_symmetric_cost_nph3_7": (
            "SR macro+beam+FS-prune (symmetric)"
        ),
        "sr_macro_beam3x2_fs_prune_one_sided_cost_nph3_7": (
            "SR macro+beam+FS-prune (one-sided)"
        ),
    }
    lines = [
        r"\begin{tabular}{@{}p{0.28\textwidth}rrrrrr@{}}",
        r"\toprule",
        r"Route & WW & IW & SW & WS & IS & SS \\",
        r"\midrule",
    ]
    for route in routes:
        cells = []
        for regime in REGIMES:
            result = route["results"][regime]
            cells.append("done" if result.get("trajectory") else "--")
        label = compact_labels.get(str(route.get("id")), str(route["label"]))
        lines.append(_tex(label) + " & " + " & ".join(cells) + r" \\")
    extra = (
        ("Historical beam 3x2, n_ph=3/7", ("--",) * 6),
        ("Phase-III batching, n_ph=3/7", ("--",) * 6),
        ("Metric-damped admission", ("--",) * 6),
    )
    for label, cells in extra:
        lines.append(_tex(label) + " & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _render_tex(report: Mapping[str, Any]) -> str:
    route_pages: list[str] = []
    for index, route in enumerate(report["routes"]):
        trajectory_name = Path(route["plots"]["trajectory"]["path"]).name
        qiskit_name = Path(route["plots"]["qiskit_target_energy"]["path"]).name
        plateau_qiskit_name = Path(route["plots"]["qiskit_plateau"]["path"]).name
        prefix = "" if index == 0 else r"\clearpage"
        route_pages.append(
            rf"""{prefix}
\twocolumn[{{%
\begin{{center}}
{{\Large\bfseries {_tex(route['label'])}}}\\[1pt]
{{\small {_tex(route['subtitle'])}}}\\[-2pt]
\includegraphics[width=0.97\textwidth]{{{trajectory_name}}}
\end{{center}}
\vspace{{-8pt}}
}}]

\section*{{Target crossing: $E_T=2\times10^{{-4}}$}}
\includegraphics[width=\columnwidth]{{{qiskit_name}}}

\begin{{center}}
\resizebox{{\columnwidth}}{{!}}{{%
{_target_energy_cost_table(route)}%
}}
\end{{center}}

\footnotesize
\textit{{Target rule:}} choose the earliest completed stored prefix with
same-cutoff $|\Delta E|\leq E_T$, where
$E_T=10^{{-4}}LE_0=2\times10^{{-4}}$. No interpolation, plateau substitution,
or terminal substitution is used. Rows that never reach the target are marked
explicitly.

\medskip
\textit{{Target cost convention:}} {_tex(route['target_energy_cost_convention'])}.
The error, active prefix, $S_{{\rm alg}}$, and Qiskit triple come from the same
exact first-crossing receipt.

\newpage

\section*{{Exact selected plateau prefix}}
\includegraphics[width=\columnwidth]{{{plateau_qiskit_name}}}

\begin{{center}}
\resizebox{{\columnwidth}}{{!}}{{%
{_plateau_cost_table(route)}%
}}
\end{{center}}

\footnotesize
\textit{{Plateau rule:}} first accepted history prefix within 10\% of the
minimum same-cutoff error over the complete stored trajectory. The red star
and dashed line in each trajectory panel mark $k_{{\rm pl}}$. Historical
30-round rows use that complete available horizon and are not relabeled as
round-50 evidence.

\medskip
\textit{{Plateau cost convention:}} {_tex(route['plateau_cost_convention'])}.
Every displayed error, active prefix, $S_{{\rm alg}}$, and Qiskit triple comes
from the same exact $k_{{\rm pl}}$ receipt.

\medskip
\textit{{Route:}} {_tex(route['policy'])}
"""
        )

    inventory_lines: list[str] = []
    unique: dict[tuple[str, str], Mapping[str, Any]] = {}
    for source in report["sources"]:
        key = (source["path"], source["sha256"])
        unique[key] = source
    for source in unique.values():
        inventory_lines.append(
            rf"\path{{{source['path']}}}\\[-1pt]"
            rf"\texttt{{sha256:{source['sha256'][:16]}...}}\par\smallskip"
        )

    route_map = {str(route["id"]): route for route in report["routes"]}
    representation_pages: list[str] = []
    for representation in ("macro", "projected_singleton"):
        comparison = report["method_representation_comparisons"][representation]
        plot_name = Path(comparison["plot"]["path"]).name
        tables: list[str] = []
        for route_id in comparison["route_ids"]:
            route = route_map[route_id]
            hit_count = comparison["hit_counts"][route_id]
            tables.append(
                rf"""\begin{{minipage}}[t]{{0.322\textwidth}}
\centering
\textbf{{{_tex(route['label'])}}}\\[-1pt]
{{\scriptsize target hits: {hit_count}/6}}\\[2pt]
\resizebox{{\linewidth}}{{!}}{{%
{_method_representation_cost_table(comparison, route_id)}%
}}
\end{{minipage}}"""
            )
        representation_label = (
            "intact macro generators"
            if representation == "macro"
            else "projected-singleton generators"
        )
        representation_pages.append(
            rf"""
\clearpage
\onecolumn
\begin{{center}}
{{\Large\bfseries Three-method comparison: {representation_label}}}\\[2pt]
{{\small SNAKE, Geo-ADAPT, and Append-ADAPT across all six regimes;
$E_T=10^{{-4}}LE_0=2\times10^{{-4}}$}}\\[5pt]
\includegraphics[width=0.97\textwidth]{{{plot_name}}}
\end{{center}}
\vspace{{-4pt}}

{tables[0]}
\hfill
{tables[1]}
\hfill
{tables[2]}

\vspace{{7pt}}
\footnotesize
Each row uses the first completed stored prefix at or below $E_T$ when that
crossing exists; otherwise it uses the validated terminal $k=50$ record.
The ``Hit'' column distinguishes those endpoints. Filled markers denote exact
first crossings and hollow markers denote terminal non-hits. All methods use
Powell with a 200-iteration cap, seed 7, the same 50-round horizon, the same
working/reference cutoff ($n_{{\rm ph}}=3$ for weak Holstein and 7 for strong
Holstein), and the same full-meta parent exposure. Qiskit resources use the
backend-free Paper-I convention (optimization level 0, transpiler seed 7,
reference-state preparation included). $S_{{\rm alg}}$ is cumulative logical
estimator work through the selected endpoint.
"""
        )

    return rf"""% MACHINE_READABLE_REPORT_JSON: {STEM}.json
% MACHINE_READABLE_REPORT_MANIFEST: report_manifest.json
\documentclass[10pt,twocolumn]{{article}}
\usepackage[margin=0.55in,columnsep=0.24in]{{geometry}}
\usepackage{{booktabs,graphicx,xcolor,hyperref,microtype}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\urlstyle{{tt}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.25em}}
\pagestyle{{plain}}
\begin{{document}}
{''.join(route_pages)}

{''.join(representation_pages)}

\clearpage
\twocolumn[{{%
\begin{{center}}
{{\Large\bfseries Remaining trajectory and cost cells}}\\[2pt]
{{\small Blank means no completed, validated artifact has been ingested.}}
\\[8pt]
\resizebox{{0.96\textwidth}}{{!}}{{%
{_status_table(report['routes'])}
}}
\end{{center}}
}}]

The pending historical-beam, Phase-III-batching, and damping rows are shown so
future results have fixed destinations. They do not represent zero error or
zero cost.

\clearpage
\raggedright
\section*{{Agent-facing source inventory}}
\footnotesize
This appendix is deliberately last. The reader-facing pages contain only
energy-error trajectories, Qiskit costs, and explicit blanks.

\tiny
{''.join(inventory_lines)}
\end{{document}}
"""


def _compile(tex_path: Path) -> Path:
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent,
        check=True,
    )
    pdf = tex_path.with_suffix(".pdf")
    if not pdf.is_file():
        raise RuntimeError("latexmk completed without producing the report PDF")
    return pdf


def _build_pool_complement_routes(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the two source-locked SNAKE pool-complement route families."""

    return [
        _build_validated_archive_route(
            route_id="sr_guarded_singleton_no_lanes_nph3_7",
            label="SR-SNAKE guarded singleton children (no lanes)",
            subtitle=(
                "50 rounds; singleton Pauli children only; physical macro lanes "
                "off; n_ph=3 weak / n_ph=7 strong"
            ),
            policy=(
                "hysteresis, ordinary novelty, beam, prune, and batching off; "
                "symmetric median/MAD arctan cost; full active-plus-singleton "
                "Phase-III response; supported-FS accepted refit"
            ),
            evidence=GUARDED_SINGLETON_EVIDENCE,
            expected_route_digest=GUARDED_SINGLETON_ROUTE_DIGEST,
            sources=sources,
            cost_convention="validated JR FakeMarrakesh, optimization level 1",
        ),
        _build_validated_archive_route(
            route_id="sr_macro_physical_lanes_nph3_7",
            label="SR-SNAKE intact macro generators (physical lanes)",
            subtitle=(
                "50 rounds; intact logical macro generators with physical lanes; "
                "n_ph=3 weak / n_ph=7 strong"
            ),
            policy=(
                "hysteresis, ordinary novelty, beam, prune, and batching off; "
                "symmetric median/MAD arctan cost; full active-plus-singleton "
                "Phase-III response; supported-FS accepted refit"
            ),
            evidence=MACRO_PHYSICAL_LANES_EVIDENCE,
            expected_route_digest=MACRO_PHYSICAL_LANES_ROUTE_DIGEST,
            sources=sources,
            cost_convention="validated JR FakeMarrakesh, optimization level 1",
        ),
    ]


def _build_pass_only_cost_arm_routes(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build only cost-arm rows that have an objective compact pass summary."""

    routes: list[dict[str, Any]] = []
    for specification in COST_ARM_ROUTE_SPECS:
        results = {
            regime: _empty_result("awaiting validation")
            for regime in REGIMES
        }
        costs = {regime: None for regime in REGIMES}
        completed = 0
        for proc, regime in enumerate(REGIMES):
            summary_path = _cost_arm_summary_path(specification, proc, regime)
            if not summary_path.is_file():
                continue
            result, cost, row_sources = _cost_arm_tracking_summary(
                summary_path=summary_path,
                regime=regime,
                expected_arm=str(specification["arm"]),
                expected_route_digest=str(
                    specification["profile_contract_sha256"]
                ),
                expected_cost_mode=str(specification["cost_mode"]),
                expected_fallback_policy=str(specification["fallback_policy"]),
            )
            results[regime] = result
            costs[regime] = cost
            sources.extend(row_sources)
            completed += 1
        if completed == 0:
            continue
        routes.append(
            {
                "id": str(specification["route_id"]),
                "label": str(specification["label"]),
                "subtitle": (
                    "50-round controller; n_ph=3 weak-Holstein / n_ph=7 "
                    "strong-Holstein; pass-only rows"
                ),
                "policy": (
                    "intact macro generators with physical lanes; historical beam "
                    "3x2 (six children maximum); live full-logical FS-trust prune; "
                    f"cost mode {specification['cost_mode']}; ordinary novelty off; "
                    "hysteresis off"
                ),
                "cost_convention": (
                    "validated JR FakeMarrakesh, optimization level 1; repaired "
                    "terminal checkpoint is the executable reporting source"
                ),
                "route_contract_sha256": str(
                    specification["profile_contract_sha256"]
                ),
                "completion_policy": (
                    "only status=pass compact summaries enter; absent/pending rows "
                    "remain explicit blanks"
                ),
                "results": results,
                "costs": costs,
            }
        )
    return routes


def _upsert_pass_only_cost_arm_routes(
    routes: Sequence[Mapping[str, Any]],
    *,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append or replace cost-arm families from pass-only compact summaries."""

    updated = json.loads(json.dumps(list(routes)))
    available = _build_pass_only_cost_arm_routes(sources)
    indices = {
        str(route.get("id")): index
        for index, route in enumerate(updated)
        if isinstance(route, Mapping)
    }
    for route in available:
        route_id = str(route["id"])
        if route_id in indices:
            updated[indices[route_id]] = route
        else:
            indices[route_id] = len(updated)
            updated.append(route)
    return updated


def _reconcile_cost_arm_pending_notes(
    notes: Sequence[Mapping[str, Any]],
    *,
    routes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep pending notes accurate when a subset of cost rows has passed."""

    route_map = {
        str(route.get("id")): route for route in routes if isinstance(route, Mapping)
    }
    reconciled: list[dict[str, Any]] = []
    for raw_note in notes:
        note = dict(raw_note)
        route = route_map.get(str(note.get("route_id")))
        results = route.get("results") if isinstance(route, Mapping) else None
        passed = [
            regime
            for regime in REGIMES
            if isinstance(results, Mapping)
            and isinstance(results.get(regime), Mapping)
            and results[regime].get("status") == "complete"
        ]
        if passed:
            pending = [regime for regime in REGIMES if regime not in passed]
            note.update(
                {
                    "status": (
                        "pass_complete" if not pending else "partial_pass_only_evidence"
                    ),
                    "passed_regimes": passed,
                    "pending_regimes": pending,
                    "display_policy": (
                        "display only rows backed by local pass summaries; all "
                        "remaining regimes stay blank"
                    ),
                }
            )
        reconciled.append(note)
    return reconciled


def _build_refreshable_comparator_routes(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rebuild comparator families whose completed archive set may grow."""

    return [
        _build_comparator_archive_route(
            route_id="append_adapt_macro_nph3_7",
            label="Append-only ADAPT macro comparator",
            subtitle="50 rounds; n_ph=3 weak-Holstein / n_ph=7 strong-Holstein",
            policy="append-only full-meta macro-generator comparator; no pruning or beam",
            archives=APPEND_MACRO_ARCHIVES,
            expected_receipt_schema="paper_i_hh_append_completion_validation_receipt_v1",
            expected_variant="append_macro",
            expected_method_id="static_full_meta_append_adapt_vqe",
            sources=sources,
            tracking_summaries=APPEND_MACRO_TRACKING_SUMMARIES,
        ),
        _build_comparator_archive_route(
            route_id="geo_adapt_projected_singleton_nph3_7",
            label="Geo-ADAPT projected-singleton comparator",
            subtitle=(
                "matched singleton child set; 50 rounds; "
                "n_ph=3 weak-Holstein / n_ph=7 strong-Holstein"
            ),
            policy=(
                "Geo-ADAPT selector over the same projected-singleton child policy "
                "used for the matched comparator study"
            ),
            archives=GEO_PROJECTED_ARCHIVES,
            expected_receipt_schema="paper_i_hh_geo_completion_validation_receipt_v1",
            expected_variant="geo_projected_singleton",
            expected_method_id="static_geo_adapt_vqe",
            sources=sources,
            tracking_summaries=GEO_PROJECTED_TRACKING_SUMMARIES,
        ),
        _build_comparator_archive_route(
            route_id="append_adapt_projected_singleton_nph3_7",
            label="Append-only ADAPT projected-singleton comparator",
            subtitle=(
                "matched singleton child set; 50 rounds; "
                "n_ph=3 weak-Holstein / n_ph=7 strong-Holstein"
            ),
            policy="append-only ADAPT over the matched projected-singleton child policy",
            archives=APPEND_PROJECTED_ARCHIVES,
            expected_receipt_schema="paper_i_hh_append_completion_validation_receipt_v1",
            expected_variant="append_projected_singleton",
            expected_method_id="static_full_meta_append_adapt_vqe",
            sources=sources,
            tracking_summaries=APPEND_PROJECTED_TRACKING_SUMMARIES,
        ),
    ]


def _refresh_late_comparator_rows_from_base(
    routes: Sequence[Mapping[str, Any]],
    *,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace only late comparator cells using compact summaries.

    All other route fields, result slots, cost slots, and source records are
    copied from the base report without reopening their transfer archives.
    """

    refreshed = json.loads(json.dumps(list(routes)))
    route_map = {
        str(route.get("id")): route
        for route in refreshed
        if isinstance(route, Mapping)
    }
    specifications = (
        (
            "append_adapt_macro_nph3_7",
            "strong_strong_u8",
            APPEND_MACRO_ARCHIVES["strong_strong_u8"],
            APPEND_MACRO_TRACKING_SUMMARIES["strong_strong_u8"],
            "paper_i_hh_append_completion_validation_receipt_v1",
            "append_macro",
            "static_full_meta_append_adapt_vqe",
        ),
        (
            "geo_adapt_projected_singleton_nph3_7",
            "weak_strong",
            GEO_PROJECTED_ARCHIVES["weak_strong"],
            GEO_PROJECTED_TRACKING_SUMMARIES["weak_strong"],
            "paper_i_hh_geo_completion_validation_receipt_v1",
            "geo_projected_singleton",
            "static_geo_adapt_vqe",
        ),
        (
            "geo_adapt_projected_singleton_nph3_7",
            "intermediate_strong",
            GEO_PROJECTED_ARCHIVES["intermediate_strong"],
            GEO_PROJECTED_TRACKING_SUMMARIES["intermediate_strong"],
            "paper_i_hh_geo_completion_validation_receipt_v1",
            "geo_projected_singleton",
            "static_geo_adapt_vqe",
        ),
        (
            "geo_adapt_projected_singleton_nph3_7",
            "strong_strong_u8",
            GEO_PROJECTED_ARCHIVES["strong_strong_u8"],
            GEO_PROJECTED_TRACKING_SUMMARIES["strong_strong_u8"],
            "paper_i_hh_geo_completion_validation_receipt_v1",
            "geo_projected_singleton",
            "static_geo_adapt_vqe",
        ),
        (
            "append_adapt_projected_singleton_nph3_7",
            "weak_strong",
            APPEND_PROJECTED_ARCHIVES["weak_strong"],
            APPEND_PROJECTED_TRACKING_SUMMARIES["weak_strong"],
            "paper_i_hh_append_completion_validation_receipt_v1",
            "append_projected_singleton",
            "static_full_meta_append_adapt_vqe",
        ),
        (
            "append_adapt_projected_singleton_nph3_7",
            "intermediate_strong",
            APPEND_PROJECTED_ARCHIVES["intermediate_strong"],
            APPEND_PROJECTED_TRACKING_SUMMARIES["intermediate_strong"],
            "paper_i_hh_append_completion_validation_receipt_v1",
            "append_projected_singleton",
            "static_full_meta_append_adapt_vqe",
        ),
        (
            "append_adapt_projected_singleton_nph3_7",
            "strong_strong_u8",
            APPEND_PROJECTED_ARCHIVES["strong_strong_u8"],
            APPEND_PROJECTED_TRACKING_SUMMARIES["strong_strong_u8"],
            "paper_i_hh_append_completion_validation_receipt_v1",
            "append_projected_singleton",
            "static_full_meta_append_adapt_vqe",
        ),
    )
    for (
        route_id,
        regime,
        archive_path,
        summary_path,
        receipt_schema,
        variant,
        method_id,
    ) in specifications:
        route = route_map.get(route_id)
        if route is None:
            raise RuntimeError(f"base report lacks late comparator route: {route_id}")
        results = route.get("results")
        costs = route.get("costs")
        if not isinstance(results, dict) or not isinstance(costs, dict):
            raise RuntimeError(f"base comparator route lacks results/costs: {route_id}")
        if regime not in results or regime not in costs:
            raise RuntimeError(
                f"base comparator route lacks late regime slot: {route_id}/{regime}"
            )
        result, cost, summary_sources = _comparator_tracking_summary(
            archive_path=archive_path,
            summary_path=summary_path,
            regime=regime,
            expected_receipt_schema=receipt_schema,
            expected_variant=variant,
            expected_method_id=method_id,
        )
        results[regime] = result
        costs[regime] = cost
        sources.extend(summary_sources)
    return refreshed


def build(
    output_dir: Path,
    *,
    plateau_json: Path = DEFAULT_PLATEAU_JSON,
    target_json: Path = DEFAULT_TARGET_JSON,
    compile_pdf: bool = True,
    inventory_only: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources: list[dict[str, Any]] = []
    source_lock_notes = _source_lock_notes(sources)
    pending_validation_notes = [dict(note) for note in PENDING_COST_ARM_NOTES]
    routes = [
        _build_whitened_route(sources),
        _build_old_off_route(sources),
        _build_archive_route(
            route_id="legacy_ordinary_novelty_on_nph2_4",
            label="Novelty ON matched control",
            subtitle="no beam, no prune, undamped; n_ph=2 weak / n_ph=4 strong",
            policy="ordinary Phase-II/III novelty multipliers on; otherwise matched to the legacy OFF control",
            archive_dir=OLD_ON_ARCHIVE_DIR,
            sources=sources,
            include_qiskit=False,
            cost_convention="no transpiled Qiskit sidecars ingested",
        ),
        _build_archive_route(
            route_id="v4_fs_prune_nph3_7",
            label="Historical pre-correction FS-prune diagnostic",
            subtitle=(
                "retained n_ph=3/7 snapshot; not the corrected "
                "hysteresis-disabled successor"
            ),
            policy="first-order Phase I; measured Phase-II curvature; full Phase-III response; FS-trust pruning enabled",
            archive_dir=V4_ARCHIVE_DIR,
            archive_overrides={"strong_weak_u8": V4_SW_ARCHIVE},
            sources=sources,
            include_qiskit=True,
            cost_convention="signed-runtime FakeMarrakesh, optimization level 1",
        ),
        _build_archive_route(
            route_id="main_noprune_symcost_nph3_7",
            label="Historical pre-correction no-prune diagnostic",
            subtitle=(
                "retained n_ph=3/7 snapshot; not the corrected "
                "hysteresis-disabled successor"
            ),
            policy="first-order Phase I; measured Phase-II curvature; full Phase-III response; supported whitening and adaptive trust",
            archive_dir=SYMCOST_ARCHIVE_DIR,
            sources=sources,
            include_qiskit=False,
            qiskit_sidecar_dir=SYMCOST_QISKIT_DIR,
            cost_convention=(
                "current JR FakeMarrakesh, optimization level 1; terminal-prefix "
                "replay passed except strong--weak, whose replay energy agreed to "
                "3.7e-15 but projective fingerprint differed (structural cost only)"
            ),
        ),
        _build_validated_archive_route(
            route_id="corrected_main_hysteresis_disabled_nph3_7",
            label="Corrected main SR (hysteresis disabled)",
            subtitle=(
                "50 rounds; ordinary novelty, beam, prune, and batching off; "
                "n_ph=3 weak / n_ph=7 strong"
            ),
            policy=(
                "symmetric median/MAD arctan cost; first-order FS-bounded Phase I; "
                "measured-required Phase-II curvature; full active-plus-singleton "
                "Phase-III response; supported whitening and adaptive trust"
            ),
            evidence=CORRECTED_MAIN_EVIDENCE,
            expected_route_digest=CORRECTED_MAIN_ROUTE_DIGEST,
            sources=sources,
            cost_convention="validated JR FakeMarrakesh, optimization level 1",
        ),
        *_build_priority_sr_routes(sources),
        _build_validated_archive_route(
            route_id="corrected_fs_prune_hysteresis_disabled_nph3_7",
            label="Corrected FS-trust prune (hysteresis disabled)",
            subtitle=(
                "50 rounds; ordinary novelty, beam, and batching off; "
                "n_ph=3 weak / n_ph=7 strong"
            ),
            policy=(
                "main SR admission model plus explicit FS-trust deletion model; "
                "measured delete-and-refit energy is the pruning authority"
            ),
            evidence=CORRECTED_FS_PRUNE_EVIDENCE,
            expected_route_digest=CORRECTED_FS_PRUNE_ROUTE_DIGEST,
            sources=sources,
            cost_convention="validated JR FakeMarrakesh, optimization level 1",
        ),
        *_build_pool_complement_routes(sources),
        *_build_pass_only_cost_arm_routes(sources),
        _build_comparator_archive_route(
            route_id="geo_adapt_macro_nph3_7",
            label="Geo-ADAPT macro comparator",
            subtitle="50 rounds; n_ph=3 weak-Holstein / n_ph=7 strong-Holstein",
            policy=(
                "full-pool projected natural-gradient selector with the preserved "
                "macro logical-generator contract"
            ),
            archives=GEO_MACRO_ARCHIVES,
            expected_receipt_schema="paper_i_hh_geo_completion_validation_receipt_v1",
            expected_variant="geo_macro",
            expected_method_id="static_geo_adapt_vqe",
            sources=sources,
        ),
        *_build_refreshable_comparator_routes(sources),
    ]
    pending_validation_notes = _reconcile_cost_arm_pending_notes(
        pending_validation_notes,
        routes=routes,
    )

    # Prefix-cost reconstruction consumes the route/source inventory, while
    # the final report consumes the resulting prefix sidecars.  This explicit
    # bootstrap surface avoids weakening either fail-closed dependency.
    if inventory_only:
        inventory_path = output_dir / f"{STEM}.json"
        inventory = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "prefix-cost bootstrap inventory for the SR tracker",
            "routes": routes,
            "source_lock_notes": source_lock_notes,
            "pending_validation_notes": pending_validation_notes,
            "sources": sources,
        }
        inventory_path.write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"json": inventory_path}

    target_payload = _attach_target_prefixes(
        routes,
        target_json=target_json.resolve(),
        sources=sources,
    )
    plateau_payload = _attach_plateau_prefixes(
        routes,
        plateau_json=plateau_json.resolve(),
        sources=sources,
    )
    sources = _latest_source_records(sources)

    generated: dict[str, Any] = {}
    for route in routes:
        trajectory = _plot_trajectory_grid(route, output_dir)
        qiskit_target = _plot_cost_grid(route, output_dir, scope="target_energy")
        qiskit_plateau = _plot_cost_grid(route, output_dir, scope="plateau")
        route["plots"] = {
            "trajectory": _generated_record(trajectory),
            "qiskit_target_energy": _generated_record(qiskit_target),
            "qiskit_plateau": _generated_record(qiskit_plateau),
        }
        generated[f"trajectory_{route['id']}"] = route["plots"]["trajectory"]
        generated[f"qiskit_target_energy_{route['id']}"] = route["plots"]["qiskit_target_energy"]
        generated[f"qiskit_plateau_{route['id']}"] = route["plots"]["qiskit_plateau"]

    comparison = _build_top_sr_append_plateau_comparison(routes)
    comparison_plot = _plot_top_sr_append_plateau_comparison(
        routes,
        comparison,
        output_dir,
    )
    comparison["plot"] = _generated_record(comparison_plot)
    generated["comparison_top2_sr_vs_append_projected_plateau"] = comparison["plot"]

    target_comparison = _build_top_sr_append_target_comparison(routes)
    target_comparison_plot = _plot_top_sr_append_plateau_comparison(
        routes,
        target_comparison,
        output_dir,
        scope="target_energy",
    )
    target_comparison["plot"] = _generated_record(target_comparison_plot)
    generated["comparison_top2_sr_vs_append_projected_target_energy"] = target_comparison["plot"]

    method_representation_comparisons: dict[str, Any] = {}
    for representation in ("macro", "projected_singleton"):
        method_comparison = _build_method_representation_comparison(
            routes,
            representation=representation,
        )
        method_comparison_plot = _plot_method_representation_comparison(
            routes,
            method_comparison,
            output_dir,
        )
        method_comparison["plot"] = _generated_record(method_comparison_plot)
        method_representation_comparisons[representation] = method_comparison
        generated[
            f"comparison_three_method_{representation}_target_or_terminal"
        ] = method_comparison["plot"]

    report = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "compact SR-SNAKE, Geo-ADAPT, and append-only ADAPT trajectory "
            "and Qiskit-cost tracker"
        ),
        "routes": routes,
        "target_energy_rule": target_payload["rule"],
        "target_energy_compile_policy": target_payload["compile_policy"],
        "target_energy_summary": target_payload["summary"],
        "plateau_rule": plateau_payload["rule"],
        "plateau_compile_policy": plateau_payload["compile_policy"],
        "plateau_summary": plateau_payload["summary"],
        "top_sr_append_plateau_comparison": comparison,
        "top_sr_append_target_energy_comparison": target_comparison,
        "method_representation_comparisons": method_representation_comparisons,
        "source_lock_notes": source_lock_notes,
        "pending_validation_notes": pending_validation_notes,
        "sources": sources,
        "generated_artifacts": generated,
    }
    tex_path = output_dir / f"{STEM}.tex"
    tex_path.write_text(_render_tex(report), encoding="utf-8")
    generated["tex"] = _generated_record(tex_path)
    pdf_path: Path | None = _compile(tex_path) if compile_pdf else None
    if pdf_path is not None:
        generated["pdf"] = _generated_record(pdf_path)
    json_path = output_dir / f"{STEM}.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generated["report_json"] = _generated_record(json_path)
    manifest_path = output_dir / "report_manifest.json"
    manifest = {
        "schema": f"{SCHEMA}_artifact_manifest",
        "created_utc": report["created_utc"],
        "route_ids": [route["id"] for route in routes],
        "source_lock_notes": source_lock_notes,
        "pending_validation_notes": pending_validation_notes,
        "consumed_artifacts": sources,
        "generated_artifacts": generated,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs = {"tex": tex_path, "json": json_path, "manifest": manifest_path}
    if pdf_path is not None:
        outputs["pdf"] = pdf_path
    return outputs


def extend_inventory_from_existing_report(
    output_dir: Path,
    *,
    base_report_json: Path,
) -> dict[str, Path]:
    """Add newly validated route families without reopening old raw archives."""

    output_dir.mkdir(parents=True, exist_ok=True)
    base = _read_json(base_report_json.resolve())
    if base.get("schema") not in {
        "paper_i_hh_sr_snake_trajectory_qiskit_tracker_v8",
        SCHEMA,
    }:
        raise RuntimeError(f"base report schema drift: {base.get('schema')!r}")
    raw_routes = base.get("routes")
    raw_sources = base.get("sources")
    if not isinstance(raw_routes, list) or not isinstance(raw_sources, list):
        raise RuntimeError("base report lacks routes or source inventory")

    sources = [dict(source) for source in raw_sources if isinstance(source, Mapping)]
    sources.append(_source_record(base_report_json.resolve()))
    raw_source_lock_notes = base.get("source_lock_notes")
    source_lock_notes = (
        json.loads(json.dumps(raw_source_lock_notes))
        if isinstance(raw_source_lock_notes, list)
        else _source_lock_notes(sources)
    )
    raw_pending_validation_notes = base.get("pending_validation_notes")
    pending_validation_notes = (
        json.loads(json.dumps(raw_pending_validation_notes))
        if isinstance(raw_pending_validation_notes, list)
        else [dict(note) for note in PENDING_COST_ARM_NOTES]
    )
    routes = _refresh_late_comparator_rows_from_base(
        [route for route in raw_routes if isinstance(route, Mapping)],
        sources=sources,
    )
    routes = _upsert_priority_sr_routes(routes, sources=sources)
    routes = _upsert_pass_only_cost_arm_routes(routes, sources=sources)
    pending_validation_notes = _reconcile_cost_arm_pending_notes(
        pending_validation_notes,
        routes=routes,
    )

    inventory_path = output_dir / f"{STEM}.json"
    inventory = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "extended prefix-cost bootstrap inventory for the SR tracker",
        "routes": routes,
        "source_lock_notes": source_lock_notes,
        "pending_validation_notes": pending_validation_notes,
        "sources": sources,
    }
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"json": inventory_path}


def build_from_existing_report(
    output_dir: Path,
    *,
    base_report_json: Path,
    plateau_json: Path,
    target_json: Path,
    compile_pdf: bool = True,
) -> dict[str, Path]:
    """Rebuild presentation layers without reopening multi-GB run archives."""

    output_dir.mkdir(parents=True, exist_ok=True)
    base = _read_json(base_report_json.resolve())
    if base.get("schema") not in {
        "paper_i_hh_sr_snake_trajectory_qiskit_tracker_v8",
        SCHEMA,
    }:
        raise RuntimeError(f"base report schema drift: {base.get('schema')!r}")
    raw_routes = base.get("routes")
    raw_sources = base.get("sources")
    if not isinstance(raw_routes, list) or not isinstance(raw_sources, list):
        raise RuntimeError("base report lacks routes or source inventory")
    routes = json.loads(json.dumps(raw_routes))
    sources = [dict(source) for source in raw_sources if isinstance(source, Mapping)]
    sources.append(_source_record(base_report_json.resolve()))
    raw_source_lock_notes = base.get("source_lock_notes")
    source_lock_notes = (
        json.loads(json.dumps(raw_source_lock_notes))
        if isinstance(raw_source_lock_notes, list)
        else _source_lock_notes(sources)
    )
    raw_pending_validation_notes = base.get("pending_validation_notes")
    pending_validation_notes = (
        json.loads(json.dumps(raw_pending_validation_notes))
        if isinstance(raw_pending_validation_notes, list)
        else [dict(note) for note in PENDING_COST_ARM_NOTES]
    )

    target_payload = _attach_target_prefixes(
        routes,
        target_json=target_json.resolve(),
        sources=sources,
    )
    plateau_payload = _attach_plateau_prefixes(
        routes,
        plateau_json=plateau_json.resolve(),
        sources=sources,
    )
    sources = _latest_source_records(sources)

    generated: dict[str, Any] = {}
    for route in routes:
        trajectory = _plot_trajectory_grid(route, output_dir)
        qiskit_target = _plot_cost_grid(route, output_dir, scope="target_energy")
        qiskit_plateau = _plot_cost_grid(route, output_dir, scope="plateau")
        route["plots"] = {
            "trajectory": _generated_record(trajectory),
            "qiskit_target_energy": _generated_record(qiskit_target),
            "qiskit_plateau": _generated_record(qiskit_plateau),
        }
        generated[f"trajectory_{route['id']}"] = route["plots"]["trajectory"]
        generated[f"qiskit_target_energy_{route['id']}"] = route["plots"]["qiskit_target_energy"]
        generated[f"qiskit_plateau_{route['id']}"] = route["plots"]["qiskit_plateau"]

    comparison = _build_top_sr_append_plateau_comparison(routes)
    comparison_plot = _plot_top_sr_append_plateau_comparison(
        routes,
        comparison,
        output_dir,
    )
    comparison["plot"] = _generated_record(comparison_plot)
    generated["comparison_top2_sr_vs_append_projected_plateau"] = comparison["plot"]

    target_comparison = _build_top_sr_append_target_comparison(routes)
    target_comparison_plot = _plot_top_sr_append_plateau_comparison(
        routes,
        target_comparison,
        output_dir,
        scope="target_energy",
    )
    target_comparison["plot"] = _generated_record(target_comparison_plot)
    generated["comparison_top2_sr_vs_append_projected_target_energy"] = target_comparison["plot"]

    method_representation_comparisons: dict[str, Any] = {}
    for representation in ("macro", "projected_singleton"):
        method_comparison = _build_method_representation_comparison(
            routes,
            representation=representation,
        )
        method_comparison_plot = _plot_method_representation_comparison(
            routes,
            method_comparison,
            output_dir,
        )
        method_comparison["plot"] = _generated_record(method_comparison_plot)
        method_representation_comparisons[representation] = method_comparison
        generated[
            f"comparison_three_method_{representation}_target_or_terminal"
        ] = method_comparison["plot"]

    report = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "compact SR-SNAKE, Geo-ADAPT, and append-only ADAPT trajectory "
            "and target/plateau Qiskit-cost tracker"
        ),
        "routes": routes,
        "target_energy_rule": target_payload["rule"],
        "target_energy_compile_policy": target_payload["compile_policy"],
        "target_energy_summary": target_payload["summary"],
        "plateau_rule": plateau_payload["rule"],
        "plateau_compile_policy": plateau_payload["compile_policy"],
        "plateau_summary": plateau_payload["summary"],
        "top_sr_append_plateau_comparison": comparison,
        "top_sr_append_target_energy_comparison": target_comparison,
        "method_representation_comparisons": method_representation_comparisons,
        "source_lock_notes": source_lock_notes,
        "pending_validation_notes": pending_validation_notes,
        "sources": sources,
        "generated_artifacts": generated,
    }
    tex_path = output_dir / f"{STEM}.tex"
    tex_path.write_text(_render_tex(report), encoding="utf-8")
    generated["tex"] = _generated_record(tex_path)
    pdf_path: Path | None = _compile(tex_path) if compile_pdf else None
    if pdf_path is not None:
        generated["pdf"] = _generated_record(pdf_path)
    json_path = output_dir / f"{STEM}.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generated["report_json"] = _generated_record(json_path)
    manifest_path = output_dir / "report_manifest.json"
    manifest = {
        "schema": f"{SCHEMA}_artifact_manifest",
        "created_utc": report["created_utc"],
        "route_ids": [route["id"] for route in routes],
        "source_lock_notes": source_lock_notes,
        "pending_validation_notes": pending_validation_notes,
        "consumed_artifacts": sources,
        "generated_artifacts": generated,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs = {"tex": tex_path, "json": json_path, "manifest": manifest_path}
    if pdf_path is not None:
        outputs["pdf"] = pdf_path
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--plateau-json", type=Path, default=DEFAULT_PLATEAU_JSON)
    parser.add_argument("--target-json", type=Path, default=DEFAULT_TARGET_JSON)
    parser.add_argument("--base-report-json", type=Path)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="write the validated route/source inventory without prefix sidecars",
    )
    args = parser.parse_args()
    if args.base_report_json is not None and args.inventory_only:
        outputs = extend_inventory_from_existing_report(
            args.output_dir.resolve(),
            base_report_json=args.base_report_json.resolve(),
        )
    elif args.base_report_json is None:
        outputs = build(
            args.output_dir.resolve(),
            plateau_json=args.plateau_json.resolve(),
            target_json=args.target_json.resolve(),
            compile_pdf=not args.no_build,
            inventory_only=args.inventory_only,
        )
    else:
        outputs = build_from_existing_report(
            args.output_dir.resolve(),
            base_report_json=args.base_report_json.resolve(),
            plateau_json=args.plateau_json.resolve(),
            target_json=args.target_json.resolve(),
            compile_pdf=not args.no_build,
        )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
