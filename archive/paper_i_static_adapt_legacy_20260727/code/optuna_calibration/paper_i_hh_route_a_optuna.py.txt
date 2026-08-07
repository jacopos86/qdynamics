#!/usr/bin/env python3
"""Six-regime Optuna adapter for the canonical Powell Paper-I SNAKE route.

The adapter deliberately keeps Optuna outside the algorithm.  One Optuna
trial defines one policy, and that policy is evaluated unchanged on all six
L=2 Hubbard--Holstein regimes through ``paper_i_hh_powell_pareto``.

Workers consume immutable cell manifests and never write to the Optuna
database.  A local planner/aggregator owns the SQLite study, which makes the
same manifests suitable for local execution or a distributed batch system.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
from typing import Any, Mapping, Sequence

from pipelines.exact_bench import paper_i_hh_powell_pareto as pareto_runner


STUDY_SCHEMA = "paper_i_hh_route_a_optuna_study_v1"
WAVE_SCHEMA = "paper_i_hh_route_a_optuna_wave_v1"
TRIAL_SCHEMA = "paper_i_hh_route_a_optuna_trial_v1"
CELL_SCHEMA = "paper_i_hh_route_a_optuna_cell_v1"
AGGREGATE_SCHEMA = "paper_i_hh_route_a_optuna_trial_aggregate_v1"
FRONT_SCHEMA = "paper_i_hh_route_a_optuna_screening_front_v1"
CHTC_EXPORT_SCHEMA = "paper_i_hh_route_a_optuna_chtc_export_v1"

OPTUNA_EXECUTION_PROFILE = "optuna_search_v1"
WAVE11_LEGAL_EXECUTION_PROFILE = "wave11_legal_fixed_policy_v1"
WAVE11_LEGAL_STUDY_NAME = "paper_i_hh_wave11_legal_six_regime_v1"
WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE = (
    "wave11_l25_adaptive_whitened_r30_v1"
)
WAVE11_ADAPTIVE_WHITENED_R30_STUDY_NAME = (
    "paper_i_hh_wave11_l25_adaptive_whitened_r30_v1"
)
JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE = (
    "jr_l10_rollback_free_whitened_r50_v1"
)
JR_L10_ROLLBACK_FREE_R50_STUDY_NAME = (
    "paper_i_hh_jr_l10_rollback_free_whitened_r50_v1"
)
REMOVED_WAVE11_R30_EXPORT = Path(
    "chtc/phase3_optuna/input/"
    "paper_i_hh_wave11_l25_adaptive_whitened_r30_20260712_v1_chtc/"
    "wave_manifest.json"
)
WAVE11_SOURCE_CHECKPOINT = Path(
    "raw_outputs/paper_i_hh_joint_selector_pareto_20260711/"
    "wave11_small_m64_m48_c128_c64_b2_l25_r15/current.json"
)
WAVE11_LEGAL_SOURCE_PLAN = Path(
    "raw_outputs/paper_i_hh_joint_selector_pareto_20260711/"
    "wave18_projected_normalized_m64_m48_c128_c64_b2_l25_uncapped_r7/plan.json"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHTC_WRAPPER = Path(
    "chtc/phase3_optuna/run_paper_i_hh_route_a_optuna_cell_apptainer.sh"
)

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak-u8",
    "weak-strong",
    "intermediate-strong",
    "strong-strong-u8",
)

OBJECTIVE_NAMES = (
    "worst_regime_abs_delta_e",
    "total_expanded_query_work",
    "total_graph_count_2q_proxy",
    "total_graph_depth_proxy",
)
OBJECTIVE_DIRECTIONS = ("minimize",) * len(OBJECTIVE_NAMES)

ALL = "all"
UNBOUNDED_SHORTLIST_CAP = 1_000_000_000
MACRO_PHASE1_VALUES: tuple[int | str, ...] = (24, 48, ALL)
MACRO_PHASE2_VALUES: tuple[int | str, ...] = (12, 24, ALL)
CHILD_PHASE1_VALUES: tuple[int | str, ...] = (32, 64, ALL)
CHILD_PHASE2_VALUES: tuple[int | str, ...] = (16, 32, 64, ALL)
BATCH_SEARCH_POOL_VALUES: tuple[int | str, ...] = (6, 10, 16, ALL)
BATCH_SIZE_CAP_VALUES = (1, 2)
ADDITIVITY_LAMBDA_VALUES = (0.1, 0.75)
BATCH_MODES = ("combinatorial", "greedy")

LOCKED_COST_WEIGHTS = {
    "2q": 0.20,
    "d": 0.20,
    "1q": 0.05,
    "theta": 0.05,
    "shot": 0.15,
}

CHTC_SOURCE_HASH_PATHS = (
    "docs/reports/pdf_utils.py",
    "pipelines/exact_bench/paper_i_hh_route_a_optuna.py",
    "pipelines/exact_bench/paper_i_hh_powell_pareto.py",
    "pipelines/exact_bench/snake_table_i_measurement_work.py",
    "pipelines/scaffold/hh_continuation_scoring.py",
    "pipelines/static_adapt/paper_i_config.py",
    "pipelines/static_adapt/paper_i_runner.py",
    "pipelines/static_adapt/builders/legal_subspace_filter.py",
    "pipelines/static_adapt/route_a_child_padding.py",
    "pipelines/static_adapt/route_a_funnel.py",
    "pipelines/static_adapt/runtime_split.py",
    "pipelines/static_adapt/route_a_shortlists.py",
    "pipelines/static_adapt/route_a_schur_selector.py",
    "pipelines/static_adapt/route_a_trust_region.py",
    "pipelines/static_adapt/joint_linear_solve.py",
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/optimizer_routes.py",
    str(CHTC_WRAPPER),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    return str(value)


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_ready(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return dict(raw)


def _manifest_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("manifest_sha256", None)
    return {**body, "manifest_sha256": _payload_hash(body)}


def _validate_manifest_hash(payload: Mapping[str, Any], *, source: Path) -> None:
    expected = payload.get("manifest_sha256")
    body = dict(payload)
    body.pop("manifest_sha256", None)
    actual = _payload_hash(body)
    if str(expected) != str(actual):
        raise ValueError(
            f"Manifest hash mismatch for {source}: expected {expected!r}, got {actual!r}."
        )


def _cap_is_all(value: int | str) -> bool:
    return str(value).strip().lower() == ALL


def _resolved_cap(value: int | str, *, zero_means_all: bool = False) -> int:
    if _cap_is_all(value):
        return 0 if zero_means_all else UNBOUNDED_SHORTLIST_CAP
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"Shortlist cap must be positive or 'all'; got {value!r}.")
    return resolved


def _validate_stage_order(
    upstream: int | str,
    downstream: int | str,
    *,
    upstream_name: str,
    downstream_name: str,
) -> None:
    # Downstream 'all' means no additional truncation of upstream survivors.
    if _cap_is_all(downstream) or _cap_is_all(upstream):
        return
    if int(downstream) > int(upstream):
        raise ValueError(
            f"{downstream_name}={downstream} exceeds {upstream_name}={upstream}."
        )


@dataclass(frozen=True)
class RouteAPolicy:
    macro_phase1_cap: int | str
    macro_phase2_cap: int | str
    child_phase1_cap: int | str
    child_phase2_cap: int | str
    batch_search_pool_size: int | str
    batch_size_cap: int
    lambda_add: float
    batch_mode: str = "combinatorial"

    def __post_init__(self) -> None:
        allowed = (
            ("macro_phase1_cap", self.macro_phase1_cap, MACRO_PHASE1_VALUES),
            ("macro_phase2_cap", self.macro_phase2_cap, MACRO_PHASE2_VALUES),
            ("child_phase1_cap", self.child_phase1_cap, CHILD_PHASE1_VALUES),
            ("child_phase2_cap", self.child_phase2_cap, CHILD_PHASE2_VALUES),
            (
                "batch_search_pool_size",
                self.batch_search_pool_size,
                BATCH_SEARCH_POOL_VALUES,
            ),
        )
        for name, value, choices in allowed:
            if value not in choices:
                raise ValueError(f"{name} must be one of {choices}; got {value!r}.")
        _validate_stage_order(
            self.macro_phase1_cap,
            self.macro_phase2_cap,
            upstream_name="macro_phase1_cap",
            downstream_name="macro_phase2_cap",
        )
        _validate_stage_order(
            self.child_phase1_cap,
            self.child_phase2_cap,
            upstream_name="child_phase1_cap",
            downstream_name="child_phase2_cap",
        )
        if int(self.batch_size_cap) not in BATCH_SIZE_CAP_VALUES:
            raise ValueError(
                f"batch_size_cap must be one of {BATCH_SIZE_CAP_VALUES}."
            )
        if float(self.lambda_add) not in ADDITIVITY_LAMBDA_VALUES:
            raise ValueError(
                f"lambda_add must be one of {ADDITIVITY_LAMBDA_VALUES}; "
                "the unpenalized zero route is excluded from this study."
            )
        if str(self.batch_mode) not in BATCH_MODES:
            raise ValueError(f"batch_mode must be one of {BATCH_MODES}.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "macro_phase1_cap": self.macro_phase1_cap,
            "macro_phase2_cap": self.macro_phase2_cap,
            "child_phase1_cap": self.child_phase1_cap,
            "child_phase2_cap": self.child_phase2_cap,
            "batch_search_pool_size": self.batch_search_pool_size,
            "batch_size_cap": int(self.batch_size_cap),
            "lambda_add": float(self.lambda_add),
            "batch_mode": str(self.batch_mode),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RouteAPolicy":
        return cls(
            macro_phase1_cap=payload["macro_phase1_cap"],
            macro_phase2_cap=payload["macro_phase2_cap"],
            child_phase1_cap=payload["child_phase1_cap"],
            child_phase2_cap=payload["child_phase2_cap"],
            batch_search_pool_size=payload["batch_search_pool_size"],
            batch_size_cap=int(payload["batch_size_cap"]),
            lambda_add=float(payload["lambda_add"]),
            batch_mode=str(payload["batch_mode"]),
        )

    def resolved_caps(self) -> dict[str, int]:
        return {
            "macro_phase1_cap": _resolved_cap(self.macro_phase1_cap),
            "macro_phase2_cap": _resolved_cap(self.macro_phase2_cap),
            "child_phase1_cap": _resolved_cap(self.child_phase1_cap),
            "child_phase2_cap": _resolved_cap(self.child_phase2_cap),
            "batch_search_pool_size": _resolved_cap(
                self.batch_search_pool_size,
                zero_means_all=True,
            ),
        }


@dataclass(frozen=True)
class Wave11LegalPolicy:
    """Fixed Wave-11 search policy with cutoff-generic legal children."""

    macro_phase1_cap: int = 64
    macro_phase2_cap: int = 48
    child_phase1_cap: int = 128
    child_phase2_cap: int = 64
    batch_search_pool_size: int = 25
    batch_size_cap: int = 2
    lambda_add: float = 0.0
    batch_mode: str = "combinatorial"

    def __post_init__(self) -> None:
        expected = {
            "macro_phase1_cap": 64,
            "macro_phase2_cap": 48,
            "child_phase1_cap": 128,
            "child_phase2_cap": 64,
            "batch_search_pool_size": 25,
            "batch_size_cap": 2,
            "lambda_add": 0.0,
            "batch_mode": "combinatorial",
        }
        actual = self.as_dict()
        if actual != expected:
            raise ValueError(
                "Wave11LegalPolicy is fixed; received drift: "
                + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "macro_phase1_cap": int(self.macro_phase1_cap),
            "macro_phase2_cap": int(self.macro_phase2_cap),
            "child_phase1_cap": int(self.child_phase1_cap),
            "child_phase2_cap": int(self.child_phase2_cap),
            "batch_search_pool_size": int(self.batch_search_pool_size),
            "batch_size_cap": int(self.batch_size_cap),
            "lambda_add": float(self.lambda_add),
            "batch_mode": str(self.batch_mode),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Wave11LegalPolicy":
        return cls(
            macro_phase1_cap=int(payload["macro_phase1_cap"]),
            macro_phase2_cap=int(payload["macro_phase2_cap"]),
            child_phase1_cap=int(payload["child_phase1_cap"]),
            child_phase2_cap=int(payload["child_phase2_cap"]),
            batch_search_pool_size=int(payload["batch_search_pool_size"]),
            batch_size_cap=int(payload["batch_size_cap"]),
            lambda_add=float(payload["lambda_add"]),
            batch_mode=str(payload["batch_mode"]),
        )

    def resolved_caps(self) -> dict[str, int]:
        return {
            "macro_phase1_cap": int(self.macro_phase1_cap),
            "macro_phase2_cap": int(self.macro_phase2_cap),
            "child_phase1_cap": int(self.child_phase1_cap),
            "child_phase2_cap": int(self.child_phase2_cap),
            "batch_search_pool_size": int(self.batch_search_pool_size),
        }


@dataclass(frozen=True)
class JrL10RollbackFreePolicy:
    """Fixed transferable JR-L10 policy for the rollback-free R50 matrix."""

    macro_phase1_cap: int = 32
    macro_phase2_cap: int = 24
    child_phase1_cap: int = 32
    child_phase2_cap: int = 25
    batch_search_pool_size: int = 10
    batch_size_cap: int = 2
    lambda_add: float = 0.0
    batch_mode: str = "combinatorial"

    def __post_init__(self) -> None:
        expected = {
            "macro_phase1_cap": 32,
            "macro_phase2_cap": 24,
            "child_phase1_cap": 32,
            "child_phase2_cap": 25,
            "batch_search_pool_size": 10,
            "batch_size_cap": 2,
            "lambda_add": 0.0,
            "batch_mode": "combinatorial",
        }
        actual = self.as_dict()
        if actual != expected:
            raise ValueError(
                "JrL10RollbackFreePolicy is fixed; received drift: "
                + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "macro_phase1_cap": int(self.macro_phase1_cap),
            "macro_phase2_cap": int(self.macro_phase2_cap),
            "child_phase1_cap": int(self.child_phase1_cap),
            "child_phase2_cap": int(self.child_phase2_cap),
            "batch_search_pool_size": int(self.batch_search_pool_size),
            "batch_size_cap": int(self.batch_size_cap),
            "lambda_add": float(self.lambda_add),
            "batch_mode": str(self.batch_mode),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "JrL10RollbackFreePolicy":
        return cls(
            macro_phase1_cap=int(payload["macro_phase1_cap"]),
            macro_phase2_cap=int(payload["macro_phase2_cap"]),
            child_phase1_cap=int(payload["child_phase1_cap"]),
            child_phase2_cap=int(payload["child_phase2_cap"]),
            batch_search_pool_size=int(payload["batch_search_pool_size"]),
            batch_size_cap=int(payload["batch_size_cap"]),
            lambda_add=float(payload["lambda_add"]),
            batch_mode=str(payload["batch_mode"]),
        )

    def resolved_caps(self) -> dict[str, int]:
        return {
            "macro_phase1_cap": int(self.macro_phase1_cap),
            "macro_phase2_cap": int(self.macro_phase2_cap),
            "child_phase1_cap": int(self.child_phase1_cap),
            "child_phase2_cap": int(self.child_phase2_cap),
            "batch_search_pool_size": int(self.batch_search_pool_size),
        }


def locked_scientific_contract(*, max_depth: int) -> dict[str, Any]:
    if int(max_depth) < 1:
        raise ValueError("max_depth must be positive.")
    return {
        "schema": "paper_i_hh_route_a_optuna_locked_contract_v1",
        "scope": "one_policy_shared_unchanged_across_all_six_L2_hh_regimes",
        "regimes": list(REGIME_ORDER),
        "num_sites": 2,
        "route": {
            "route_id": "route_a",
            "funnel_mode": "child_12_joint_schur_v1",
            "phase0_enabled": False,
            "macro_phase3_enabled": False,
            "child_phase3_enabled": False,
            "macro_stages": ["macro_phase1", "macro_phase2"],
            "child_stages": ["child_phase1", "child_phase2"],
            "final_selection_authority": "joint_ansatz_plus_batch_schur",
            "child_identity_policy": "global_pauli_word_v1",
            "child_identity_normalization": (
                "projective_normalized_pauli_polynomial_v1"
            ),
            "pauli_word_subset_sizes": [1],
            "child_symmetry_policy": "hard_guard",
            "allow_repeats": True,
        },
        "selector": {
            "joint_batch_context_mode": "full_ansatz_v1",
            "geometry_mode": "full_residual_gram_hessian_v1",
            "additivity_policy": "soft_penalty_v1",
            "allowed_lambda_add": list(ADDITIVITY_LAMBDA_VALUES),
            "allowed_batch_size_cap": list(BATCH_SIZE_CAP_VALUES),
            "allowed_batch_modes": list(BATCH_MODES),
            "cost_enabled": True,
            "cost_weights": dict(LOCKED_COST_WEIGHTS),
        },
        "beam": {
            "enabled": True,
            "live_branches": 3,
            "children_per_parent": 2,
            "legacy_lambda_affects_decisions": False,
        },
        "pruning": {
            "enabled": True,
            "policy": "recoverability_ladder_v1",
            "deletion_authority": "remove_refit_energy_safety",
        },
        "optimizer": {
            "name": "POWELL",
            "maxiter": 50,
            "maxfev": 200,
            "seed": 7,
            "reoptimization_policy": "full",
            "full_refit_every": 1,
            "final_full_refit": True,
            "full_insertion_position_search": True,
        },
        "execution": {
            "max_controller_rounds": int(max_depth),
            "state_backend": "compiled",
            "noise": "off",
            "qiskit_costing": "post_run_finalists_only",
        },
        "optuna_variables_only": [
            "macro_phase1_cap",
            "macro_phase2_cap",
            "child_phase1_cap",
            "child_phase2_cap",
            "batch_search_pool_size",
            "batch_size_cap",
            "lambda_add",
        ],
        "objective_names": list(OBJECTIVE_NAMES),
        "accuracy_secondary_diagnostic": "arithmetic_mean_abs_delta_e",
    }


def wave11_legal_scientific_contract(*, max_depth: int) -> dict[str, Any]:
    """Return the immutable six-regime legal rerun contract for Wave 11."""

    if int(max_depth) != 15:
        raise ValueError("The Wave-11 legal matrix is fixed at 15 controller rounds.")
    source_records: dict[str, dict[str, str]] = {}
    for role, relative in (
        ("wave11_unchecked_checkpoint", WAVE11_SOURCE_CHECKPOINT),
        ("legal_projection_reference_plan", WAVE11_LEGAL_SOURCE_PLAN),
    ):
        source = REPO_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing Wave-11 provenance source: {source}")
        source_records[role] = {
            "path": str(relative),
            "sha256": _sha256_file(source),
        }
    return {
        "schema": "paper_i_hh_wave11_legal_fixed_contract_v1",
        "run_class": "candidate_fixed_policy_matrix",
        "scope": "wave11_settings_cutoff_generic_legal_children_all_six_L2_hh_regimes",
        "regimes": list(REGIME_ORDER),
        "cutoff_contract": {
            "weak_holstein": {"n_ph_work": 2, "n_ph_ref": 2},
            "strong_holstein": {"n_ph_work": 4, "n_ph_ref": 4},
        },
        "num_sites": 2,
        "policy": Wave11LegalPolicy().as_dict(),
        "route": {
            "route_id": "route_a",
            "funnel_mode": "child_12_joint_schur_v1",
            "phase0_enabled": False,
            "macro_phase3_enabled": False,
            "child_phase3_enabled": False,
            "child_padding_policy": "exact_projected_grouped_v1",
            "child_padding_semantics": (
                "pre_score_exact_projection_then_global_projected_identity_dedup_v1"
            ),
            "child_identity_policy": "global_pauli_word_v1",
            "child_identity_normalization": (
                "projective_normalized_pauli_polynomial_v1"
            ),
            "pauli_word_subset_sizes": [1],
            "child_symmetry_policy": "hard_guard",
            "allow_repeats": True,
        },
        "selector": {
            "joint_batch_context_mode": "full_ansatz_v1",
            "geometry_mode": "full_residual_gram_hessian_v1",
            "batch_mode": "combinatorial_reduced_plane",
            "batch_size_cap": 2,
            "batch_search_pool_size": 25,
            "batch_search_feasibility_policy": "raw_ranked_legacy_v1",
            "additivity_policy": "soft_penalty_v1",
            "lambda_add": 0.0,
            "cost_enabled": True,
            "cost_weights": dict(LOCKED_COST_WEIGHTS),
        },
        "beam": {"enabled": True, "live_branches": 3, "children_per_parent": 2},
        "pruning": {
            "enabled": True,
            "policy": "recoverability_ladder_v1",
            "deletion_authority": "remove_refit_energy_safety",
        },
        "optimizer": {
            "name": "POWELL",
            "maxiter": 50,
            "maxfev": None,
            "seed": 7,
            "reoptimization_policy": "full",
            "full_refit_every": 1,
            "final_full_refit": True,
            "full_insertion_position_search": True,
        },
        "execution": {
            "max_controller_rounds": 15,
            "state_backend": "compiled",
            "noise": "off",
            "result_payload_mode": "summary_checkpoint_v1",
            "current_checkpoint_every_controller_round": True,
            "query_accounting": [
                "winning_lineage_S_alg",
                "all_expanded_branch_query_work_diagnostic",
            ],
            "qiskit_costing": "post_run_exact_prefix",
        },
        "source_records": source_records,
    }


def _wave11_legal_contract_hash(*, max_depth: int) -> str:
    return _payload_hash(
        {"wave11_legal_scientific_contract": wave11_legal_scientific_contract(max_depth=max_depth)}
    )


def wave11_adaptive_whitened_r30_scientific_contract(
    *, max_depth: int
) -> dict[str, Any]:
    """Return the approved R30 extension of the legal Wave-11 L25 policy."""

    if int(max_depth) != 30:
        raise ValueError("The adaptive-whitened Wave-11 matrix is fixed at 30 rounds.")
    contract = json.loads(
        json.dumps(wave11_legal_scientific_contract(max_depth=15))
    )
    contract["schema"] = "paper_i_hh_wave11_l25_adaptive_whitened_r30_contract_v1"
    contract["scope"] = (
        "fresh_wave11_L25_policy_all_six_L2_hh_regimes_adaptive_whitened_r30"
    )
    contract["execution"]["max_controller_rounds"] = 30
    contract["selector"].update(
        {
            "joint_linear_solve_policy": "supported_metric_whitened_eigh_v1",
            "coordinate_scope": "full_active_ansatz_plus_batch_v1",
            "trust_region_update_policy": "displacement_calibrated_unbounded_v2",
            "trust_region_initial_radius": 0.25,
            "trust_region_scientific_radius_min": 0.0,
            "trust_region_scientific_radius_max": None,
            "trust_region_rate_limiter": "none",
            "exhaustion_retry_policy": "expand_all_then_force_singleton_v1",
            "exhaustion_retry_telemetry_required": True,
        }
    )
    contract["approved_changes_vs_wave11_legal_fixed_policy_v1"] = [
        "execution.max_controller_rounds:15->30",
        "selector.joint_linear_solve_policy:source_implicit->supported_metric_whitened_eigh_v1",
        "selector.trust_region_update_policy:fixed->displacement_calibrated_unbounded_v2",
        "selector.exhaustion_retry_policy:stop->expand_all_then_force_singleton_v1",
    ]
    contract["fresh_start"] = True
    return contract


def _wave11_adaptive_whitened_r30_contract_hash(*, max_depth: int) -> str:
    return _payload_hash(
        {
            "wave11_adaptive_whitened_r30_scientific_contract": (
                wave11_adaptive_whitened_r30_scientific_contract(
                    max_depth=max_depth
                )
            )
        }
    )


def jr_l10_rollback_free_r50_scientific_contract(
    *, max_depth: int
) -> dict[str, Any]:
    if int(max_depth) != 50:
        raise ValueError("The rollback-free JR-L10 matrix is fixed at 50 rounds.")
    return {
        "schema": "paper_i_hh_jr_l10_rollback_free_r50_contract_v1",
        "run_class": "candidate_fixed_policy_matrix",
        "scope": "fresh_transferable_jr_l10_policy_all_six_L2_hh_regimes",
        "regimes": list(REGIME_ORDER),
        "cutoff_contract": {
            "weak_holstein": {"n_ph_work": 2, "n_ph_ref": 2},
            "strong_holstein": {"n_ph_work": 4, "n_ph_ref": 4},
        },
        "num_sites": 2,
        "policy": JrL10RollbackFreePolicy().as_dict(),
        "route": {
            "route_id": "route_a",
            "funnel_mode": "child_12_joint_response_v2",
            "phase0_enabled": False,
            "macro_phase3_enabled": False,
            "child_phase3_enabled": False,
            "child_padding_policy_by_sector": {
                "weak_holstein_nph2": "nph2_exact_projected_grouped_v1",
                "strong_holstein": "exact_projected_grouped_v1",
            },
            "child_padding_semantics": (
                "pre_score_exact_projection_then_global_projected_identity_dedup_v1"
            ),
            "child_identity_policy": "global_pauli_word_v1",
            "child_identity_normalization": (
                "projective_normalized_pauli_polynomial_v1"
            ),
            "pauli_word_subset_sizes": [1],
            "child_symmetry_policy": "hard_guard",
            "allow_repeats": True,
            "structural_rollback_control": "absent",
            "duplicate_cooldown_control": "absent",
        },
        "selector": {
            "joint_batch_context_mode": "full_ansatz_v1",
            "geometry_mode": "full_residual_gram_hessian_v1",
            "batch_mode": "combinatorial_reduced_plane",
            "batch_size_cap": 2,
            "batch_search_pool_size": 10,
            "batch_search_feasibility_policy": "joint_subset_gate_v1",
            "additivity_policy": "soft_penalty_v1",
            "lambda_add": 0.0,
            "joint_linear_solve_policy": "supported_metric_whitened_eigh_v1",
            "joint_step_warm_start_mode": "exact_applied_joint_step_guarded_v1",
            "trust_region_update_policy": "displacement_calibrated_unbounded_v2",
            "trust_region_initial_radius": 0.25,
            "trust_region_scientific_radius_min": 0.0,
            "trust_region_scientific_radius_max": None,
            "trust_region_rate_limiter": "none",
            "selector_exhaustion_retry_policy": "stop",
            "cost_enabled": True,
            "cost_weights": dict(LOCKED_COST_WEIGHTS),
        },
        "beam": {
            "enabled": True,
            "live_branches": 3,
            "children_per_parent": 2,
            "legacy_lambda_affects_decisions": False,
        },
        "pruning": {
            "enabled": True,
            "policy": "recoverability_ladder_v1",
            "deletion_authority": "remove_refit_energy_safety",
        },
        "optimizer": {
            "name": "POWELL",
            "maxiter": 50,
            "maxfev": 200,
            "seed": 7,
            "reoptimization_policy": "full",
            "full_refit_every": 1,
            "final_full_refit": True,
            "full_insertion_position_search": True,
        },
        "execution": {
            "fresh_start": True,
            "max_controller_rounds": 50,
            "state_backend": "compiled",
            "noise": "off",
            "result_payload_mode": "summary_checkpoint_v1",
            "current_checkpoint_every_controller_round": True,
            "query_accounting": [
                "winning_lineage_S_alg",
                "discarded_branch_search_work_diagnostic_separate",
            ],
            "qiskit_costing": "post_run_exact_prefix",
        },
        "supersedes_removed_cluster": {
            "cluster_id": 8775666,
            "batch_name": (
                "paper-i-hh-wave11-l25-adaptive-whitened-r30-20260712-v1"
            ),
            "local_wave_manifest": str(REMOVED_WAVE11_R30_EXPORT),
        },
    }


def _jr_l10_rollback_free_r50_contract_hash(*, max_depth: int) -> str:
    return _payload_hash(
        {
            "jr_l10_rollback_free_r50_scientific_contract": (
                jr_l10_rollback_free_r50_scientific_contract(
                    max_depth=max_depth
                )
            )
        }
    )


def _import_optuna() -> Any:
    try:
        import optuna  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Optuna is required for this adapter.") from exc
    return optuna


def _storage_url(path: Path) -> str:
    return f"sqlite:///{path.expanduser().resolve().as_posix()}"


def _study_contract_hash(*, max_depth: int, batch_mode: str) -> str:
    return _payload_hash(
        {
            "locked_scientific_contract": locked_scientific_contract(
                max_depth=max_depth
            ),
            "study_batch_mode": str(batch_mode),
        }
    )


def _load_or_create_study(
    *,
    database: Path,
    study_name: str,
    max_depth: int,
    batch_mode: str,
    sampler_seed: int,
    population_size: int,
) -> Any:
    if str(batch_mode) not in BATCH_MODES:
        raise ValueError(f"batch_mode must be one of {BATCH_MODES}.")
    optuna = _import_optuna()
    database.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.NSGAIISampler(
        population_size=int(population_size),
        seed=int(sampler_seed),
    )
    study = optuna.create_study(
        study_name=str(study_name),
        storage=_storage_url(database),
        directions=list(OBJECTIVE_DIRECTIONS),
        sampler=sampler,
        load_if_exists=True,
    )
    expected_hash = _study_contract_hash(
        max_depth=int(max_depth),
        batch_mode=str(batch_mode),
    )
    existing_hash = study.user_attrs.get("scientific_contract_hash")
    if existing_hash is None:
        study.set_user_attr("schema", STUDY_SCHEMA)
        study.set_user_attr("scientific_contract_hash", expected_hash)
        study.set_user_attr(
            "locked_scientific_contract",
            locked_scientific_contract(max_depth=int(max_depth)),
        )
        study.set_user_attr("study_batch_mode", str(batch_mode))
        study.set_user_attr("objective_names", list(OBJECTIVE_NAMES))
    elif str(existing_hash) != str(expected_hash):
        raise ValueError(
            "Existing Optuna study has a different scientific contract; "
            "create a new study instead of mixing policies."
        )
    return study


def suggest_policy(trial: Any, *, batch_mode: str) -> RouteAPolicy:
    policy = RouteAPolicy(
        macro_phase1_cap=trial.suggest_categorical(
            "macro_phase1_cap", list(MACRO_PHASE1_VALUES)
        ),
        macro_phase2_cap=trial.suggest_categorical(
            "macro_phase2_cap", list(MACRO_PHASE2_VALUES)
        ),
        child_phase1_cap=trial.suggest_categorical(
            "child_phase1_cap", list(CHILD_PHASE1_VALUES)
        ),
        child_phase2_cap=trial.suggest_categorical(
            "child_phase2_cap", list(CHILD_PHASE2_VALUES)
        ),
        batch_search_pool_size=trial.suggest_categorical(
            "batch_search_pool_size", list(BATCH_SEARCH_POOL_VALUES)
        ),
        batch_size_cap=int(
            trial.suggest_categorical(
                "batch_size_cap", list(BATCH_SIZE_CAP_VALUES)
            )
        ),
        lambda_add=float(
            trial.suggest_categorical(
                "lambda_add", list(ADDITIVITY_LAMBDA_VALUES)
            )
        ),
        batch_mode=str(batch_mode),
    )
    return policy


def build_pareto_argv(
    policy: RouteAPolicy | Wave11LegalPolicy | JrL10RollbackFreePolicy,
    *,
    regime: str,
    output_json: Path,
    current_json: Path,
    max_depth: int,
    dry_run: bool,
    gradient_workers: int = 1,
    beam_parent_workers: int = 1,
    runtime_split_child_workers: int = 0,
    joint_pair_workers: int = 1,
) -> list[str]:
    if str(regime) not in REGIME_ORDER:
        raise ValueError(f"Unknown L=2 HH regime: {regime!r}.")
    caps = policy.resolved_caps()
    argv = [
        "run",
        "--regime",
        str(regime),
        "--output-json",
        str(output_json),
        "--current-json",
        str(current_json),
        "--max-depth",
        str(int(max_depth)),
        "--optimizer-maxiter",
        "50",
        "--scipy-maxfev",
        "200",
        "--seed",
        "7",
        "--geometry-window-size",
        "99",
        "--subset-sizes",
        "1",
        "--child-symmetry-policy",
        "hard_guard",
        "--batch-mode",
        str(policy.batch_mode),
        "--batch-target-size",
        str(int(policy.batch_size_cap)),
        "--batch-size-cap",
        str(int(policy.batch_size_cap)),
        "--batch-search-pool-size",
        str(int(caps["batch_search_pool_size"])),
        "--batch-additivity-policy",
        "soft_penalty_v1",
        "--batch-additivity-lambda",
        str(float(policy.lambda_add)),
        "--joint-batch-context-mode",
        "full_ansatz_v1",
        "--beam-width",
        "3",
        "--beam-children-per-parent",
        "2",
        # These caps are inactive but remain positive for the typed runner.
        "--phase0-shortlist-size",
        "1",
        "--phase1-shortlist-size",
        str(int(caps["macro_phase1_cap"])),
        "--phase2-shortlist-size",
        str(int(caps["macro_phase2_cap"])),
        "--phase2-shortlist-fraction",
        "1.0",
        "--child-phase1-shortlist-size",
        str(int(caps["child_phase1_cap"])),
        "--child-phase2-shortlist-size",
        str(int(caps["child_phase2_cap"])),
        "--child-phase3-shortlist-size",
        "1",
        "--funnel-mode",
        "child_12_joint_schur_v1",
        "--physical-lane-shortlist-aggressiveness",
        "3",
        "--gradient-workers",
        str(int(gradient_workers)),
        "--beam-parent-workers",
        str(int(beam_parent_workers)),
        "--runtime-split-child-workers",
        str(int(runtime_split_child_workers)),
        "--joint-pair-workers",
        str(int(joint_pair_workers)),
        "--result-payload-mode",
        "summary_checkpoint_v1",
    ]
    if bool(dry_run):
        argv.append("--dry-run")
    return argv


def build_wave11_legal_pareto_argv(
    policy: Wave11LegalPolicy,
    *,
    regime: str,
    output_json: Path,
    current_json: Path,
    max_depth: int,
    dry_run: bool,
    gradient_workers: int = 1,
    beam_parent_workers: int = 1,
    runtime_split_child_workers: int = 0,
    joint_pair_workers: int = 1,
) -> list[str]:
    """Build the Wave-11 command with the legal child repair as the sole semantic delta."""

    argv = build_pareto_argv(
        policy,
        regime=regime,
        output_json=output_json,
        current_json=current_json,
        max_depth=max_depth,
        dry_run=dry_run,
        gradient_workers=gradient_workers,
        beam_parent_workers=beam_parent_workers,
        runtime_split_child_workers=runtime_split_child_workers,
        joint_pair_workers=joint_pair_workers,
    )
    maxfev_index = argv.index("--scipy-maxfev") + 1
    argv[maxfev_index] = "0"
    phase0_index = argv.index("--phase0-shortlist-size") + 1
    argv[phase0_index] = "256"
    child_phase3_index = argv.index("--child-phase3-shortlist-size") + 1
    argv[child_phase3_index] = "4096"
    argv.extend(
        [
            "--child-padding-policy",
            "exact_projected_grouped_v1",
            "--batch-search-feasibility-policy",
            "raw_ranked_legacy_v1",
        ]
    )
    return argv


def build_wave11_adaptive_whitened_r30_pareto_argv(
    policy: Wave11LegalPolicy,
    *,
    regime: str,
    output_json: Path,
    current_json: Path,
    max_depth: int,
    dry_run: bool,
    gradient_workers: int = 1,
    beam_parent_workers: int = 1,
    runtime_split_child_workers: int = 0,
    joint_pair_workers: int = 1,
) -> list[str]:
    """Build the fresh R30 L25 route with explicit whitening/retry controls."""

    if int(max_depth) != 30:
        raise ValueError("Adaptive-whitened Wave-11 commands require 30 rounds.")
    argv = build_wave11_legal_pareto_argv(
        policy,
        regime=regime,
        output_json=output_json,
        current_json=current_json,
        max_depth=max_depth,
        dry_run=dry_run,
        gradient_workers=gradient_workers,
        beam_parent_workers=beam_parent_workers,
        runtime_split_child_workers=runtime_split_child_workers,
        joint_pair_workers=joint_pair_workers,
    )
    argv.extend(
        [
            "--joint-linear-solve-policy",
            "supported_metric_whitened_eigh_v1",
            "--trust-region-update-policy",
            "displacement_calibrated_unbounded_v2",
            "--trust-region-radius-min",
            "0",
            "--joint-step-warm-start-mode",
            "off",
            "--selector-exhaustion-retry-policy",
            "expand_all_then_force_singleton_v1",
        ]
    )
    return argv


def build_jr_l10_rollback_free_r50_pareto_argv(
    policy: JrL10RollbackFreePolicy,
    *,
    regime: str,
    output_json: Path,
    current_json: Path,
    max_depth: int,
    dry_run: bool,
    gradient_workers: int = 1,
    beam_parent_workers: int = 1,
    runtime_split_child_workers: int = 0,
    joint_pair_workers: int = 1,
) -> list[str]:
    if int(max_depth) != 50:
        raise ValueError("Rollback-free JR-L10 commands require 50 rounds.")
    argv = build_pareto_argv(
        policy,
        regime=regime,
        output_json=output_json,
        current_json=current_json,
        max_depth=max_depth,
        dry_run=dry_run,
        gradient_workers=gradient_workers,
        beam_parent_workers=beam_parent_workers,
        runtime_split_child_workers=runtime_split_child_workers,
        joint_pair_workers=joint_pair_workers,
    )
    argv[argv.index("--funnel-mode") + 1] = "child_12_joint_response_v2"
    argv[argv.index("--phase0-shortlist-size") + 1] = "256"
    argv[argv.index("--child-phase3-shortlist-size") + 1] = "4096"
    child_padding_policy = (
        "nph2_exact_projected_grouped_v1"
        if str(regime) in {"weak-weak", "intermediate-weak", "strong-weak-u8"}
        else "exact_projected_grouped_v1"
    )
    argv.extend(
        [
            "--child-padding-policy",
            child_padding_policy,
            "--batch-search-feasibility-policy",
            "joint_subset_gate_v1",
            "--joint-linear-solve-policy",
            "supported_metric_whitened_eigh_v1",
            "--joint-step-warm-start-mode",
            "exact_applied_joint_step_guarded_v1",
            "--trust-region-update-policy",
            "displacement_calibrated_unbounded_v2",
            "--trust-region-radius-min",
            "0",
            "--selector-exhaustion-retry-policy",
            "stop",
        ]
    )
    return argv


def _expect_equal(
    checks: list[dict[str, Any]],
    *,
    path: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual != expected:
        checks.append({"path": str(path), "expected": expected, "actual": actual})


def validate_locked_pareto_plan(
    plan: Mapping[str, Any],
    *,
    policy: RouteAPolicy | Wave11LegalPolicy | JrL10RollbackFreePolicy,
    max_depth: int,
    expected_scipy_maxfev: int = 200,
    expected_child_padding_policy: str | None = None,
    expected_batch_search_feasibility_policy: str | None = None,
    expected_joint_linear_solve_policy: str | None = None,
    expected_trust_region_update_policy: str | None = None,
    expected_selector_exhaustion_retry_policy: str | None = None,
    expected_funnel_mode: str = "child_12_joint_schur_v1",
    expected_joint_step_warm_start_mode: str | None = None,
    require_rollback_controls_absent: bool = False,
) -> dict[str, Any]:
    scientific = plan.get("scientific_settings")
    if not isinstance(scientific, Mapping):
        raise ValueError("Powell plan is missing scientific_settings.")
    invocation = scientific.get("route_a_invocation")
    run_kwargs = scientific.get("run_kwargs")
    if not isinstance(invocation, Mapping) or not isinstance(run_kwargs, Mapping):
        raise ValueError("Powell plan is missing Route-A invocation details.")
    mechanisms = invocation.get("mechanisms")
    shortlists = invocation.get("shortlists")
    if not isinstance(mechanisms, Mapping) or not isinstance(shortlists, Mapping):
        raise ValueError("Powell plan is missing mechanism or shortlist details.")
    funnel = mechanisms.get("route_a_funnel")
    if not isinstance(funnel, Mapping):
        raise ValueError("Powell plan is missing the typed Route-A funnel.")
    caps = policy.resolved_caps()
    mismatches: list[dict[str, Any]] = []
    for path, actual, expected in (
        ("profile", invocation.get("profile"), "canonical"),
        ("route_id", invocation.get("route_id"), "route_a"),
        ("pool_key", invocation.get("pool_key"), "full_meta"),
        ("max_adapt_iterations", invocation.get("max_adapt_iterations"), int(max_depth)),
        ("shortlists.phase0_enabled", shortlists.get("phase0_enabled"), False),
        ("shortlists.child_phase3_enabled", shortlists.get("child_phase3_enabled"), False),
        ("shortlists.phase1_size", shortlists.get("phase1_size"), caps["macro_phase1_cap"]),
        ("shortlists.phase2_size", shortlists.get("phase2_size"), caps["macro_phase2_cap"]),
        ("shortlists.child_phase1_size", shortlists.get("child_phase1_size"), caps["child_phase1_cap"]),
        ("shortlists.child_phase2_size", shortlists.get("child_phase2_size"), caps["child_phase2_cap"]),
        ("mechanisms.cost_enabled", mechanisms.get("cost_enabled"), True),
        ("mechanisms.cost_weights", mechanisms.get("cost_weights"), LOCKED_COST_WEIGHTS),
        ("mechanisms.beam_enabled", mechanisms.get("beam_enabled"), True),
        ("mechanisms.pruning_enabled", mechanisms.get("pruning_enabled"), True),
        ("mechanisms.pauli_child_pool_enabled", mechanisms.get("pauli_child_pool_enabled"), True),
        ("mechanisms.pauli_word_subset_sizes", mechanisms.get("pauli_word_subset_sizes"), [1]),
        ("mechanisms.child_symmetry_policy", mechanisms.get("child_symmetry_policy"), "hard_guard"),
        ("mechanisms.child_identity_normalization", mechanisms.get("child_identity_normalization"), "projective_normalized_pauli_polynomial_v1"),
        ("mechanisms.final_selection_authority", mechanisms.get("final_selection_authority"), "joint_ansatz_plus_batch_schur"),
        ("mechanisms.batch_selection_mode", mechanisms.get("batch_selection_mode"), f"{policy.batch_mode}_reduced_plane"),
        ("mechanisms.batch_size_cap", mechanisms.get("batch_size_cap"), int(policy.batch_size_cap)),
        ("mechanisms.batch_search_pool_size", mechanisms.get("batch_search_pool_size"), caps["batch_search_pool_size"]),
        ("mechanisms.batch_additivity_policy", mechanisms.get("batch_additivity_policy"), "soft_penalty_v1"),
        ("mechanisms.batch_additivity_lambda", mechanisms.get("batch_additivity_lambda"), float(policy.lambda_add)),
        ("mechanisms.joint_batch_context_mode", mechanisms.get("joint_batch_context_mode"), "full_ansatz_v1"),
        ("funnel.mode", funnel.get("mode"), str(expected_funnel_mode)),
        ("funnel.phase0_policy", funnel.get("phase0_policy"), "disabled"),
        ("funnel.child_identity_policy", funnel.get("child_identity_policy"), "global_pauli_word_v1"),
        ("run_kwargs.allow_repeats", run_kwargs.get("allow_repeats"), True),
        ("run_kwargs.adapt_state_backend", run_kwargs.get("adapt_state_backend"), "compiled"),
        ("run_kwargs.adapt_reopt_policy", run_kwargs.get("adapt_reopt_policy"), "full"),
        ("run_kwargs.adapt_full_refit_every", run_kwargs.get("adapt_full_refit_every"), 1),
        ("run_kwargs.adapt_final_full_refit", run_kwargs.get("adapt_final_full_refit"), True),
        ("run_kwargs.adapt_insertion_mode", run_kwargs.get("adapt_insertion_mode"), "always"),
        ("run_kwargs.phase0_pilot_enabled", run_kwargs.get("phase0_pilot_enabled"), False),
        ("run_kwargs.phase1_prune_enabled", run_kwargs.get("phase1_prune_enabled"), True),
        ("run_kwargs.adapt_beam_live_branches", run_kwargs.get("adapt_beam_live_branches"), 3),
        ("run_kwargs.adapt_beam_children_per_parent", run_kwargs.get("adapt_beam_children_per_parent"), 2),
        ("run_kwargs.adapt_beam_lambda", run_kwargs.get("adapt_beam_lambda"), 0.0),
        ("run_kwargs.adapt_analytic_noise_std", run_kwargs.get("adapt_analytic_noise_std"), 0.0),
        ("run_kwargs.phase3_oracle_gradient_config", run_kwargs.get("phase3_oracle_gradient_config"), None),
        ("run_kwargs.final_noise_audit_config", run_kwargs.get("final_noise_audit_config"), None),
        ("run_kwargs.maxiter", run_kwargs.get("maxiter"), 50),
        (
            "run_kwargs.adapt_scipy_maxfev",
            run_kwargs.get("adapt_scipy_maxfev"),
            int(expected_scipy_maxfev),
        ),
        ("run_kwargs.seed", run_kwargs.get("seed"), 7),
    ):
        _expect_equal(
            mismatches,
            path=path,
            actual=actual,
            expected=expected,
        )
    if expected_child_padding_policy is not None:
        child_padding = funnel.get("child_padding")
        child_padding_map = (
            dict(child_padding) if isinstance(child_padding, Mapping) else {}
        )
        _expect_equal(
            mismatches,
            path="mechanisms.child_padding_policy",
            actual=mechanisms.get("child_padding_policy"),
            expected=str(expected_child_padding_policy),
        )
        _expect_equal(
            mismatches,
            path="funnel.child_padding.policy",
            actual=child_padding_map.get("policy"),
            expected=str(expected_child_padding_policy),
        )
        _expect_equal(
            mismatches,
            path="mechanisms.child_padding_guard_semantics",
            actual=mechanisms.get("child_padding_guard_semantics"),
            expected=(
                "pre_score_exact_projection_then_global_projected_identity_dedup_v1"
            ),
        )
    if expected_batch_search_feasibility_policy is not None:
        schur_selector = funnel.get("schur_selector")
        schur_selector_map = (
            dict(schur_selector)
            if isinstance(schur_selector, Mapping)
            else {}
        )
        _expect_equal(
            mismatches,
            path="funnel.schur_selector.batch_search_feasibility_policy",
            actual=schur_selector_map.get("batch_search_feasibility_policy"),
            expected=str(expected_batch_search_feasibility_policy),
        )
    schur_selector_raw = funnel.get("schur_selector")
    schur_selector_map = (
        dict(schur_selector_raw)
        if isinstance(schur_selector_raw, Mapping)
        else {}
    )
    if expected_joint_linear_solve_policy is not None:
        _expect_equal(
            mismatches,
            path="funnel.schur_selector.joint_linear_solve_policy",
            actual=schur_selector_map.get("joint_linear_solve_policy"),
            expected=str(expected_joint_linear_solve_policy),
        )
    if expected_trust_region_update_policy is not None:
        trust_update = schur_selector_map.get("trust_region_update")
        trust_update_map = (
            dict(trust_update) if isinstance(trust_update, Mapping) else {}
        )
        _expect_equal(
            mismatches,
            path="funnel.schur_selector.trust_region_update.policy",
            actual=trust_update_map.get("policy"),
            expected=str(expected_trust_region_update_policy),
        )
        _expect_equal(
            mismatches,
            path="funnel.schur_selector.trust_region_update.scientific_radius_min_effective",
            actual=trust_update_map.get("scientific_radius_min_effective"),
            expected=0.0,
        )
        _expect_equal(
            mismatches,
            path="funnel.schur_selector.trust_region_update.scientific_radius_max_effective",
            actual=trust_update_map.get("scientific_radius_max_effective"),
            expected=None,
        )
    if expected_selector_exhaustion_retry_policy is not None:
        _expect_equal(
            mismatches,
            path="funnel.schur_selector.exhaustion_retry_policy",
            actual=schur_selector_map.get("exhaustion_retry_policy"),
            expected=str(expected_selector_exhaustion_retry_policy),
        )
    if expected_joint_step_warm_start_mode is not None:
        mechanism_warm_start = mechanisms.get("joint_step_warm_start")
        mechanism_warm_start_map = (
            dict(mechanism_warm_start)
            if isinstance(mechanism_warm_start, Mapping)
            else {}
        )
        funnel_warm_start = funnel.get("joint_step_warm_start")
        funnel_warm_start_map = (
            dict(funnel_warm_start)
            if isinstance(funnel_warm_start, Mapping)
            else {}
        )
        for path, actual in (
            (
                "mechanisms.joint_step_warm_start.mode",
                mechanism_warm_start_map.get("mode"),
            ),
            (
                "funnel.joint_step_warm_start.mode",
                funnel_warm_start_map.get("mode"),
            ),
        ):
            _expect_equal(
                mismatches,
                path=path,
                actual=actual,
                expected=str(expected_joint_step_warm_start_mode),
            )
    if require_rollback_controls_absent:
        forbidden = {
            "run_kwargs.adapt_rollback_mode": "adapt_rollback_mode" in run_kwargs,
            "run_kwargs.adapt_rollback_tolerance": (
                "adapt_rollback_tolerance" in run_kwargs
            ),
            "funnel.duplicate_cooldown_policy": (
                "duplicate_cooldown_policy" in funnel
            ),
        }
        for field, present in forbidden.items():
            if present:
                mismatches.append(
                    {"path": field, "expected": "absent", "actual": "present"}
                )
    if mismatches:
        raise ValueError(
            "Route-A Optuna locked-contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "schema": "paper_i_hh_route_a_optuna_plan_validation_v1",
        "status": "pass",
        "scientific_settings_hash": plan.get("scientific_settings_hash"),
        "locked_field_count": int(
            41
            + (expected_joint_linear_solve_policy is not None)
            + 3 * (expected_trust_region_update_policy is not None)
            + (expected_selector_exhaustion_retry_policy is not None)
            + 2 * (expected_joint_step_warm_start_mode is not None)
            + 3 * bool(require_rollback_controls_absent)
        ),
    }


def _cell_id(trial_number: int, regime: str) -> str:
    return f"trial_{int(trial_number):06d}__{str(regime).replace('-', '_')}"


def _write_cell_manifest(
    *,
    output_dir: Path,
    study_name: str,
    trial_number: int,
    policy: RouteAPolicy | Wave11LegalPolicy,
    regime: str,
    max_depth: int,
    scientific_contract_hash: str,
    execution_profile: str = OPTUNA_EXECUTION_PROFILE,
    run_class: str = "diagnostic_pareto_search",
) -> tuple[Path, dict[str, Any]]:
    cell_id = _cell_id(trial_number, regime)
    output_relative_dir = Path(f"trial_{int(trial_number):06d}") / str(regime)
    payload = _manifest_with_hash(
        {
            "schema": CELL_SCHEMA,
            "generated_utc": _utc_now(),
            "run_class": str(run_class),
            "execution_profile": str(execution_profile),
            "study_name": str(study_name),
            "trial_number": int(trial_number),
            "cell_id": str(cell_id),
            "regime": str(regime),
            "num_sites": 2,
            "policy": policy.as_dict(),
            "policy_sha256": _payload_hash(policy.as_dict()),
            "scientific_contract_hash": str(scientific_contract_hash),
            "max_depth": int(max_depth),
            "output_relative_dir": str(output_relative_dir),
            "expected_outputs": [
                "result.json",
                "current.json",
                "compact_summary.json",
                "query_work_sidecar.json",
            ],
        }
    )
    manifest_path = output_dir / "cell_manifests" / f"{cell_id}.json"
    _write_json(manifest_path, payload)
    return manifest_path, payload


def plan_wave11_legal_matrix(
    *,
    output_dir: Path,
    max_depth: int = 15,
) -> dict[str, Any]:
    """Write one immutable fixed-policy trial spanning all six HH regimes."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Wave output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = wave11_legal_scientific_contract(max_depth=int(max_depth))
    contract_hash = _wave11_legal_contract_hash(max_depth=int(max_depth))
    policy = Wave11LegalPolicy()
    trial_number = 0
    cells: list[dict[str, Any]] = []
    cell_records: list[tuple[str, Path]] = []
    for regime in REGIME_ORDER:
        manifest_path, cell = _write_cell_manifest(
            output_dir=output_dir,
            study_name=WAVE11_LEGAL_STUDY_NAME,
            trial_number=trial_number,
            policy=policy,
            regime=str(regime),
            max_depth=int(max_depth),
            scientific_contract_hash=contract_hash,
            execution_profile=WAVE11_LEGAL_EXECUTION_PROFILE,
            run_class="candidate_fixed_policy_matrix",
        )
        relative_manifest = manifest_path.relative_to(output_dir)
        cells.append(
            {
                "cell_id": str(cell["cell_id"]),
                "regime": str(regime),
                "manifest": str(relative_manifest),
                "output_relative_dir": str(cell["output_relative_dir"]),
            }
        )
        cell_records.append((str(cell["cell_id"]), relative_manifest))
    trial_payload = _manifest_with_hash(
        {
            "schema": TRIAL_SCHEMA,
            "generated_utc": _utc_now(),
            "run_class": "candidate_fixed_policy_matrix",
            "execution_profile": WAVE11_LEGAL_EXECUTION_PROFILE,
            "study_name": WAVE11_LEGAL_STUDY_NAME,
            "trial_number": trial_number,
            "policy": policy.as_dict(),
            "policy_sha256": _payload_hash(policy.as_dict()),
            "scientific_contract_hash": contract_hash,
            "regime_order": list(REGIME_ORDER),
            "cell_count": len(cells),
            "cells": cells,
        }
    )
    trial_manifest_path = output_dir / "trial_manifests" / "trial_000000.json"
    _write_json(trial_manifest_path, trial_payload)
    records_path = output_dir / "cell_records.tsv"
    records_path.write_text(
        "cell_id\tcell_manifest\n"
        + "".join(f"{cell_id}\t{path}\n" for cell_id, path in cell_records),
        encoding="utf-8",
    )
    ids_path = output_dir / "cell_ids.txt"
    ids_path.write_text(
        "".join(f"{cell_id}\n" for cell_id, _ in cell_records),
        encoding="utf-8",
    )
    wave = _manifest_with_hash(
        {
            "schema": WAVE_SCHEMA,
            "generated_utc": _utc_now(),
            "run_class": "candidate_fixed_policy_matrix",
            "execution_profile": WAVE11_LEGAL_EXECUTION_PROFILE,
            "study_name": WAVE11_LEGAL_STUDY_NAME,
            "study_database": None,
            "study_storage_policy": "no_optuna_fixed_immutable_policy",
            "scientific_contract": contract,
            "scientific_contract_hash": contract_hash,
            "batch_mode": policy.batch_mode,
            "objective_names": list(OBJECTIVE_NAMES),
            "objective_directions": list(OBJECTIVE_DIRECTIONS),
            "trial_count": 1,
            "cell_count": len(cell_records),
            "trials": [
                {
                    "trial_number": trial_number,
                    "policy": policy.as_dict(),
                    "policy_sha256": _payload_hash(policy.as_dict()),
                    "trial_manifest": str(trial_manifest_path.relative_to(output_dir)),
                    "cell_count": len(cells),
                }
            ],
            "cell_records_tsv": records_path.name,
            "cell_ids_txt": ids_path.name,
            "result_root": "results",
        }
    )
    _write_json(output_dir / "wave_manifest.json", wave)
    return wave


def plan_wave11_adaptive_whitened_r30_matrix(
    *, output_dir: Path, max_depth: int = 30
) -> dict[str, Any]:
    """Write the fresh six-regime R30 adaptive/whitened L25 matrix."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Wave output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = wave11_adaptive_whitened_r30_scientific_contract(
        max_depth=int(max_depth)
    )
    contract_hash = _wave11_adaptive_whitened_r30_contract_hash(
        max_depth=int(max_depth)
    )
    policy = Wave11LegalPolicy()
    cells: list[dict[str, Any]] = []
    cell_records: list[tuple[str, Path]] = []
    for regime in REGIME_ORDER:
        manifest_path, cell = _write_cell_manifest(
            output_dir=output_dir,
            study_name=WAVE11_ADAPTIVE_WHITENED_R30_STUDY_NAME,
            trial_number=0,
            policy=policy,
            regime=str(regime),
            max_depth=int(max_depth),
            scientific_contract_hash=contract_hash,
            execution_profile=WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
            run_class="candidate_fixed_policy_matrix",
        )
        relative_manifest = manifest_path.relative_to(output_dir)
        cells.append(
            {
                "cell_id": str(cell["cell_id"]),
                "regime": str(regime),
                "manifest": str(relative_manifest),
                "output_relative_dir": str(cell["output_relative_dir"]),
            }
        )
        cell_records.append((str(cell["cell_id"]), relative_manifest))
    trial_payload = _manifest_with_hash(
        {
            "schema": TRIAL_SCHEMA,
            "generated_utc": _utc_now(),
            "run_class": "candidate_fixed_policy_matrix",
            "execution_profile": WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
            "study_name": WAVE11_ADAPTIVE_WHITENED_R30_STUDY_NAME,
            "trial_number": 0,
            "policy": policy.as_dict(),
            "policy_sha256": _payload_hash(policy.as_dict()),
            "scientific_contract_hash": contract_hash,
            "regime_order": list(REGIME_ORDER),
            "cell_count": len(cells),
            "cells": cells,
        }
    )
    trial_manifest_path = output_dir / "trial_manifests/trial_000000.json"
    _write_json(trial_manifest_path, trial_payload)
    records_path = output_dir / "cell_records.tsv"
    records_path.write_text(
        "cell_id\tcell_manifest\n"
        + "".join(f"{cell_id}\t{path}\n" for cell_id, path in cell_records),
        encoding="utf-8",
    )
    ids_path = output_dir / "cell_ids.txt"
    ids_path.write_text(
        "".join(f"{cell_id}\n" for cell_id, _ in cell_records),
        encoding="utf-8",
    )
    wave = _manifest_with_hash(
        {
            "schema": WAVE_SCHEMA,
            "generated_utc": _utc_now(),
            "run_class": "candidate_fixed_policy_matrix",
            "execution_profile": WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
            "study_name": WAVE11_ADAPTIVE_WHITENED_R30_STUDY_NAME,
            "study_database": None,
            "study_storage_policy": "no_optuna_fixed_immutable_policy",
            "scientific_contract": contract,
            "scientific_contract_hash": contract_hash,
            "batch_mode": policy.batch_mode,
            "objective_names": list(OBJECTIVE_NAMES),
            "objective_directions": list(OBJECTIVE_DIRECTIONS),
            "trial_count": 1,
            "cell_count": len(cell_records),
            "trials": [
                {
                    "trial_number": 0,
                    "policy": policy.as_dict(),
                    "policy_sha256": _payload_hash(policy.as_dict()),
                    "trial_manifest": str(
                        trial_manifest_path.relative_to(output_dir)
                    ),
                    "cell_count": len(cells),
                }
            ],
            "cell_records_tsv": records_path.name,
            "cell_ids_txt": ids_path.name,
            "result_root": "results",
        }
    )
    _write_json(output_dir / "wave_manifest.json", wave)
    return wave


def _flatten_settings(value: Any, *, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_settings(value[key], prefix=child_prefix))
        return flattened
    return {prefix: _json_ready(value)}


def _settings_diff(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[dict[str, Any]]:
    left = _flatten_settings(before)
    right = _flatten_settings(after)
    return [
        {"path": path, "before": left.get(path), "after": right.get(path)}
        for path in sorted(set(left) | set(right))
        if left.get(path) != right.get(path)
    ]


def plan_jr_l10_rollback_free_r50_matrix(
    *, output_dir: Path, max_depth: int = 50
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Wave output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = jr_l10_rollback_free_r50_scientific_contract(
        max_depth=int(max_depth)
    )
    contract_hash = _jr_l10_rollback_free_r50_contract_hash(
        max_depth=int(max_depth)
    )
    policy = JrL10RollbackFreePolicy()
    cells: list[dict[str, Any]] = []
    cell_records: list[tuple[str, Path]] = []
    for regime in REGIME_ORDER:
        manifest_path, cell = _write_cell_manifest(
            output_dir=output_dir,
            study_name=JR_L10_ROLLBACK_FREE_R50_STUDY_NAME,
            trial_number=0,
            policy=policy,
            regime=str(regime),
            max_depth=int(max_depth),
            scientific_contract_hash=contract_hash,
            execution_profile=JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE,
            run_class="candidate_fixed_policy_matrix",
        )
        relative_manifest = manifest_path.relative_to(output_dir)
        cells.append(
            {
                "cell_id": str(cell["cell_id"]),
                "regime": str(regime),
                "manifest": str(relative_manifest),
                "output_relative_dir": str(cell["output_relative_dir"]),
            }
        )
        cell_records.append((str(cell["cell_id"]), relative_manifest))
    trial_payload = _manifest_with_hash(
        {
            "schema": TRIAL_SCHEMA,
            "generated_utc": _utc_now(),
            "run_class": "candidate_fixed_policy_matrix",
            "execution_profile": JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE,
            "study_name": JR_L10_ROLLBACK_FREE_R50_STUDY_NAME,
            "trial_number": 0,
            "policy": policy.as_dict(),
            "policy_sha256": _payload_hash(policy.as_dict()),
            "scientific_contract_hash": contract_hash,
            "regime_order": list(REGIME_ORDER),
            "cell_count": len(cells),
            "cells": cells,
        }
    )
    trial_manifest_path = output_dir / "trial_manifests/trial_000000.json"
    _write_json(trial_manifest_path, trial_payload)
    records_path = output_dir / "cell_records.tsv"
    records_path.write_text(
        "cell_id\tcell_manifest\n"
        + "".join(f"{cell_id}\t{path}\n" for cell_id, path in cell_records),
        encoding="utf-8",
    )
    ids_path = output_dir / "cell_ids.txt"
    ids_path.write_text(
        "".join(f"{cell_id}\n" for cell_id, _ in cell_records),
        encoding="utf-8",
    )
    removed_wave_path = REPO_ROOT / REMOVED_WAVE11_R30_EXPORT
    if not removed_wave_path.is_file():
        raise FileNotFoundError(removed_wave_path)
    removed_wave = _read_json(removed_wave_path)
    removed_contract = removed_wave.get("scientific_contract")
    if not isinstance(removed_contract, Mapping):
        raise ValueError("Removed cluster wave manifest lacks scientific_contract.")
    settings_diff = _settings_diff(removed_contract, contract)
    diff_payload = _manifest_with_hash(
        {
            "schema": "paper_i_hh_jr_l10_rollback_free_settings_diff_v1",
            "generated_utc": _utc_now(),
            "status": "pass",
            "removed_cluster_id": 8775666,
            "removed_wave_manifest": str(REMOVED_WAVE11_R30_EXPORT),
            "removed_wave_manifest_sha256": _sha256_file(removed_wave_path),
            "new_execution_profile": JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE,
            "new_scientific_contract_hash": contract_hash,
            "changed_field_count": len(settings_diff),
            "changes": settings_diff,
            "critical_contract": {
                "fresh_start": True,
                "max_controller_rounds": 50,
                "policy": policy.as_dict(),
                "funnel_mode": "child_12_joint_response_v2",
                "batch_search_feasibility_policy": "joint_subset_gate_v1",
                "joint_linear_solve_policy": "supported_metric_whitened_eigh_v1",
                "joint_step_warm_start_mode": "exact_applied_joint_step_guarded_v1",
                "trust_region_update_policy": "displacement_calibrated_unbounded_v2",
                "selector_exhaustion_retry_policy": "stop",
                "powell_maxiter": 50,
                "powell_maxfev": 200,
                "structural_rollback_control": "absent",
                "duplicate_cooldown_control": "absent",
            },
        }
    )
    diff_path = output_dir / "settings_diff_vs_removed_cluster_8775666.json"
    _write_json(diff_path, diff_payload)
    wave = _manifest_with_hash(
        {
            "schema": WAVE_SCHEMA,
            "generated_utc": _utc_now(),
            "run_class": "candidate_fixed_policy_matrix",
            "execution_profile": JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE,
            "study_name": JR_L10_ROLLBACK_FREE_R50_STUDY_NAME,
            "study_database": None,
            "study_storage_policy": "no_optuna_fixed_immutable_policy",
            "scientific_contract": contract,
            "scientific_contract_hash": contract_hash,
            "settings_diff": str(diff_path.relative_to(output_dir)),
            "settings_diff_sha256": _sha256_file(diff_path),
            "batch_mode": policy.batch_mode,
            "objective_names": list(OBJECTIVE_NAMES),
            "objective_directions": list(OBJECTIVE_DIRECTIONS),
            "trial_count": 1,
            "cell_count": len(cell_records),
            "trials": [
                {
                    "trial_number": 0,
                    "policy": policy.as_dict(),
                    "policy_sha256": _payload_hash(policy.as_dict()),
                    "trial_manifest": str(
                        trial_manifest_path.relative_to(output_dir)
                    ),
                    "cell_count": len(cells),
                }
            ],
            "cell_records_tsv": records_path.name,
            "cell_ids_txt": ids_path.name,
            "result_root": "results",
        }
    )
    _write_json(output_dir / "wave_manifest.json", wave)
    return wave


def plan_wave(
    *,
    database: Path,
    study_name: str,
    output_dir: Path,
    n_trials: int,
    max_depth: int,
    batch_mode: str,
    sampler_seed: int,
    population_size: int,
) -> dict[str, Any]:
    if int(n_trials) < 1:
        raise ValueError("n_trials must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Wave output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    study = _load_or_create_study(
        database=database,
        study_name=study_name,
        max_depth=int(max_depth),
        batch_mode=str(batch_mode),
        sampler_seed=int(sampler_seed),
        population_size=int(population_size),
    )
    contract = locked_scientific_contract(max_depth=int(max_depth))
    contract_hash = str(study.user_attrs["scientific_contract_hash"])
    trials: list[dict[str, Any]] = []
    cell_records: list[tuple[str, Path]] = []
    optuna = _import_optuna()
    while len(trials) < int(n_trials):
        trial = study.ask()
        try:
            policy = suggest_policy(trial, batch_mode=str(batch_mode))
        except ValueError as exc:
            trial.set_user_attr("invalid_policy_reason", str(exc))
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        trial.set_user_attr("policy", policy.as_dict())
        trial.set_user_attr("policy_sha256", _payload_hash(policy.as_dict()))
        trial.set_user_attr("scientific_contract_hash", contract_hash)
        trial.set_user_attr("regimes", list(REGIME_ORDER))
        cells: list[dict[str, Any]] = []
        for regime in REGIME_ORDER:
            manifest_path, cell = _write_cell_manifest(
                output_dir=output_dir,
                study_name=str(study_name),
                trial_number=int(trial.number),
                policy=policy,
                regime=str(regime),
                max_depth=int(max_depth),
                scientific_contract_hash=contract_hash,
            )
            relative_manifest = manifest_path.relative_to(output_dir)
            cells.append(
                {
                    "cell_id": str(cell["cell_id"]),
                    "regime": str(regime),
                    "manifest": str(relative_manifest),
                    "output_relative_dir": str(cell["output_relative_dir"]),
                }
            )
            cell_records.append((str(cell["cell_id"]), relative_manifest))
        trial_payload = _manifest_with_hash(
            {
                "schema": TRIAL_SCHEMA,
                "generated_utc": _utc_now(),
                "study_name": str(study_name),
                "trial_number": int(trial.number),
                "policy": policy.as_dict(),
                "policy_sha256": _payload_hash(policy.as_dict()),
                "scientific_contract_hash": contract_hash,
                "regime_order": list(REGIME_ORDER),
                "cell_count": len(cells),
                "cells": cells,
            }
        )
        trial_manifest_path = (
            output_dir / "trial_manifests" / f"trial_{int(trial.number):06d}.json"
        )
        _write_json(trial_manifest_path, trial_payload)
        trial.set_user_attr(
            "trial_manifest",
            str(trial_manifest_path.relative_to(output_dir)),
        )
        trials.append(
            {
                "trial_number": int(trial.number),
                "policy": policy.as_dict(),
                "policy_sha256": _payload_hash(policy.as_dict()),
                "trial_manifest": str(trial_manifest_path.relative_to(output_dir)),
                "cell_count": len(cells),
            }
        )
    records_path = output_dir / "cell_records.tsv"
    records_path.write_text(
        "cell_id\tcell_manifest\n"
        + "".join(f"{cell_id}\t{path}\n" for cell_id, path in cell_records),
        encoding="utf-8",
    )
    ids_path = output_dir / "cell_ids.txt"
    ids_path.write_text(
        "".join(f"{cell_id}\n" for cell_id, _ in cell_records),
        encoding="utf-8",
    )
    wave = _manifest_with_hash(
        {
            "schema": WAVE_SCHEMA,
            "generated_utc": _utc_now(),
            "run_class": "diagnostic_pareto_search",
            "execution_profile": OPTUNA_EXECUTION_PROFILE,
            "study_name": str(study_name),
            "study_database": str(database.expanduser().resolve()),
            "study_storage_policy": (
                "planner_and_aggregator_only_workers_use_immutable_manifests"
            ),
            "scientific_contract": contract,
            "scientific_contract_hash": contract_hash,
            "batch_mode": str(batch_mode),
            "objective_names": list(OBJECTIVE_NAMES),
            "objective_directions": list(OBJECTIVE_DIRECTIONS),
            "trial_count": len(trials),
            "cell_count": len(cell_records),
            "trials": trials,
            "cell_records_tsv": records_path.name,
            "cell_ids_txt": ids_path.name,
            "result_root": "results",
        }
    )
    _write_json(output_dir / "wave_manifest.json", wave)
    return wave


def _cell_output_paths(
    cell: Mapping[str, Any],
    *,
    output_root: Path,
) -> tuple[Path, Path]:
    output_dir = output_root / str(cell["output_relative_dir"])
    return output_dir / "result.json", output_dir / "current.json"


def run_cell(
    *,
    cell_manifest: Path,
    output_root: Path,
    dry_run: bool,
    gradient_workers: int,
    beam_parent_workers: int,
    runtime_split_child_workers: int,
    joint_pair_workers: int,
) -> dict[str, Any]:
    cell = _read_json(cell_manifest)
    _validate_manifest_hash(cell, source=cell_manifest)
    if str(cell.get("schema")) != CELL_SCHEMA:
        raise ValueError(f"Unexpected cell schema: {cell.get('schema')!r}.")
    if str(cell.get("regime")) not in REGIME_ORDER:
        raise ValueError(f"Unexpected cell regime: {cell.get('regime')!r}.")
    policy_raw = cell.get("policy")
    if not isinstance(policy_raw, Mapping):
        raise ValueError("Cell manifest has no policy object.")
    execution_profile = str(
        cell.get("execution_profile", OPTUNA_EXECUTION_PROFILE)
    )
    if execution_profile in {
        WAVE11_LEGAL_EXECUTION_PROFILE,
        WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
    }:
        policy: RouteAPolicy | Wave11LegalPolicy | JrL10RollbackFreePolicy = Wave11LegalPolicy.from_mapping(
            policy_raw
        )
    elif execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE:
        policy = JrL10RollbackFreePolicy.from_mapping(policy_raw)
    elif execution_profile == OPTUNA_EXECUTION_PROFILE:
        policy = RouteAPolicy.from_mapping(policy_raw)
    else:
        raise ValueError(f"Unexpected cell execution profile: {execution_profile!r}.")
    if str(cell.get("policy_sha256")) != _payload_hash(policy.as_dict()):
        raise ValueError("Cell policy hash does not match its policy object.")
    if execution_profile == WAVE11_LEGAL_EXECUTION_PROFILE:
        expected_contract_hash = _wave11_legal_contract_hash(
            max_depth=int(cell["max_depth"])
        )
    elif execution_profile == WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE:
        expected_contract_hash = _wave11_adaptive_whitened_r30_contract_hash(
            max_depth=int(cell["max_depth"])
        )
    elif execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE:
        expected_contract_hash = _jr_l10_rollback_free_r50_contract_hash(
            max_depth=int(cell["max_depth"])
        )
    else:
        expected_contract_hash = _study_contract_hash(
            max_depth=int(cell["max_depth"]),
            batch_mode=str(policy.batch_mode),
        )
    if str(cell.get("scientific_contract_hash")) != expected_contract_hash:
        raise ValueError("Cell scientific contract hash is stale or inconsistent.")
    result_json, current_json = _cell_output_paths(cell, output_root=output_root)
    result_json.parent.mkdir(parents=True, exist_ok=True)
    argv_builder = {
        WAVE11_LEGAL_EXECUTION_PROFILE: build_wave11_legal_pareto_argv,
        WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE: (
            build_wave11_adaptive_whitened_r30_pareto_argv
        ),
        JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE: (
            build_jr_l10_rollback_free_r50_pareto_argv
        ),
    }.get(execution_profile, build_pareto_argv)
    argv = argv_builder(
        policy,
        regime=str(cell["regime"]),
        output_json=result_json,
        current_json=current_json,
        max_depth=int(cell["max_depth"]),
        dry_run=True,
        gradient_workers=int(gradient_workers),
        beam_parent_workers=int(beam_parent_workers),
        runtime_split_child_workers=int(runtime_split_child_workers),
        joint_pair_workers=int(joint_pair_workers),
    )
    plan_args = pareto_runner.build_parser().parse_args(argv)
    plan = pareto_runner.run_once(plan_args)
    validation = validate_locked_pareto_plan(
        plan,
        policy=policy,
        max_depth=int(cell["max_depth"]),
        expected_scipy_maxfev=(
            0
            if execution_profile
            in {
                WAVE11_LEGAL_EXECUTION_PROFILE,
                WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
            }
            else 200
        ),
        expected_child_padding_policy=(
            (
                "nph2_exact_projected_grouped_v1"
                if str(cell["regime"])
                in {"weak-weak", "intermediate-weak", "strong-weak-u8"}
                else "exact_projected_grouped_v1"
            )
            if execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE
            else
            "exact_projected_grouped_v1"
            if execution_profile
            in {
                WAVE11_LEGAL_EXECUTION_PROFILE,
                WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
            }
            else None
        ),
        expected_batch_search_feasibility_policy=(
            "joint_subset_gate_v1"
            if execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE
            else "raw_ranked_legacy_v1"
            if execution_profile
            in {
                WAVE11_LEGAL_EXECUTION_PROFILE,
                WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
            }
            else None
        ),
        expected_joint_linear_solve_policy=(
            "supported_metric_whitened_eigh_v1"
            if execution_profile
            in {
                WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
                JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE,
            }
            else None
        ),
        expected_trust_region_update_policy=(
            "displacement_calibrated_unbounded_v2"
            if execution_profile
            in {
                WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
                JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE,
            }
            else None
        ),
        expected_selector_exhaustion_retry_policy=(
            "stop"
            if execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE
            else "expand_all_then_force_singleton_v1"
            if execution_profile
            == WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE
            else None
        ),
        expected_funnel_mode=(
            "child_12_joint_response_v2"
            if execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE
            else "child_12_joint_schur_v1"
        ),
        expected_joint_step_warm_start_mode=(
            "exact_applied_joint_step_guarded_v1"
            if execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE
            else None
        ),
        require_rollback_controls_absent=(
            execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE
        ),
    )
    if bool(dry_run):
        output = {
            "schema": "paper_i_hh_route_a_optuna_cell_plan_v1",
            "status": "planned",
            "cell_manifest": str(cell_manifest),
            "cell_manifest_sha256": cell.get("manifest_sha256"),
            "cell": cell,
            "locked_contract_validation": validation,
            "pareto_plan": plan,
        }
        _write_json(result_json, output)
        return output
    run_argv = [token for token in argv if token != "--dry-run"]
    run_args = pareto_runner.build_parser().parse_args(run_argv)
    result = pareto_runner.run_once(run_args)
    validate_locked_pareto_plan(
        result,
        policy=policy,
        max_depth=int(cell["max_depth"]),
        expected_scipy_maxfev=(
            0
            if execution_profile
            in {
                WAVE11_LEGAL_EXECUTION_PROFILE,
                WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
            }
            else 200
        ),
        expected_child_padding_policy=(
            (
                "nph2_exact_projected_grouped_v1"
                if str(cell["regime"])
                in {"weak-weak", "intermediate-weak", "strong-weak-u8"}
                else "exact_projected_grouped_v1"
            )
            if execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE
            else
            "exact_projected_grouped_v1"
            if execution_profile
            in {
                WAVE11_LEGAL_EXECUTION_PROFILE,
                WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
            }
            else None
        ),
        expected_batch_search_feasibility_policy=(
            "joint_subset_gate_v1"
            if execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE
            else "raw_ranked_legacy_v1"
            if execution_profile
            in {
                WAVE11_LEGAL_EXECUTION_PROFILE,
                WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
            }
            else None
        ),
        expected_joint_linear_solve_policy=(
            "supported_metric_whitened_eigh_v1"
            if execution_profile
            in {
                WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
                JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE,
            }
            else None
        ),
        expected_trust_region_update_policy=(
            "displacement_calibrated_unbounded_v2"
            if execution_profile
            in {
                WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE,
                JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE,
            }
            else None
        ),
        expected_selector_exhaustion_retry_policy=(
            "stop"
            if execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE
            else "expand_all_then_force_singleton_v1"
            if execution_profile
            == WAVE11_ADAPTIVE_WHITENED_R30_EXECUTION_PROFILE
            else None
        ),
        expected_funnel_mode=(
            "child_12_joint_response_v2"
            if execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE
            else "child_12_joint_schur_v1"
        ),
        expected_joint_step_warm_start_mode=(
            "exact_applied_joint_step_guarded_v1"
            if execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE
            else None
        ),
        require_rollback_controls_absent=(
            execution_profile == JR_L10_ROLLBACK_FREE_R50_EXECUTION_PROFILE
        ),
    )
    return result


def _finite_nonnegative(value: Any, *, field: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {field}: {value!r}.") from exc
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative; got {value!r}.")
    return resolved


def aggregate_trial_results(
    *,
    trial_manifest: Path,
    wave_root: Path,
    result_root: Path,
) -> dict[str, Any]:
    trial = _read_json(trial_manifest)
    _validate_manifest_hash(trial, source=trial_manifest)
    if str(trial.get("schema")) != TRIAL_SCHEMA:
        raise ValueError(f"Unexpected trial schema: {trial.get('schema')!r}.")
    policy_raw = trial.get("policy")
    if not isinstance(policy_raw, Mapping):
        raise ValueError("Trial manifest has no policy object.")
    policy = RouteAPolicy.from_mapping(policy_raw)
    cells_raw = trial.get("cells")
    if not isinstance(cells_raw, Sequence):
        raise ValueError("Trial manifest has no cells array.")
    cells = [dict(cell) for cell in cells_raw if isinstance(cell, Mapping)]
    regimes = [str(cell.get("regime")) for cell in cells]
    if tuple(regimes) != REGIME_ORDER:
        raise ValueError(
            f"Trial must contain all six regimes in canonical order; got {regimes}."
        )
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[str] = []
    for cell_ref in cells:
        cell_manifest_path = wave_root / str(cell_ref["manifest"])
        cell = _read_json(cell_manifest_path)
        _validate_manifest_hash(cell, source=cell_manifest_path)
        result_json, _ = _cell_output_paths(cell, output_root=result_root)
        if not result_json.is_file():
            missing.append(str(cell["cell_id"]))
            continue
        result = _read_json(result_json)
        if str(result.get("status")) != "complete":
            failed.append(str(cell["cell_id"]))
            continue
        summary = result.get("summary")
        if not isinstance(summary, Mapping) or not bool(summary.get("success")):
            failed.append(str(cell["cell_id"]))
            continue
        scientific = result.get("scientific_settings")
        regime_payload = (
            scientific.get("regime") if isinstance(scientific, Mapping) else None
        )
        if not isinstance(regime_payload, Mapping) or str(
            regime_payload.get("name")
        ) != str(cell["regime"]):
            raise ValueError(
                f"Result regime mismatch for cell {cell['cell_id']!r}."
            )
        validate_locked_pareto_plan(
            result,
            policy=policy,
            max_depth=int(cell["max_depth"]),
        )
        graph_proxy = summary.get("graph_cost_proxy")
        if not isinstance(graph_proxy, Mapping):
            raise ValueError(f"Cell {cell['cell_id']!r} has no graph cost proxy.")
        graph_proxy = dict(graph_proxy)
        raw_graph_count_2q = graph_proxy.get("count_2q")
        raw_graph_depth = graph_proxy.get("depth")
        graph_proxy_fallback_used = False
        if raw_graph_count_2q is None and raw_graph_depth is None:
            run_kwargs = (
                scientific.get("run_kwargs")
                if isinstance(scientific, Mapping)
                else None
            )
            backend_cost_mode = (
                run_kwargs.get("phase3_backend_cost_mode")
                if isinstance(run_kwargs, Mapping)
                else None
            )
            if str(backend_cost_mode) != "proxy":
                raise ValueError(
                    f"Cell {cell['cell_id']!r} has null graph proxies outside "
                    "the known proxy-mode reconstruction contract."
                )
            adapt_payload = result.get("adapt_vqe")
            if not isinstance(adapt_payload, Mapping):
                raise ValueError(
                    f"Cell {cell['cell_id']!r} has no adapt_vqe payload for "
                    "terminal graph-proxy reconstruction."
                )
            graph_proxy = pareto_runner.terminal_internal_graph_proxy_from_payload(
                adapt_payload
            )
            graph_proxy_fallback_used = True
        elif (raw_graph_count_2q is None) != (raw_graph_depth is None):
            raise ValueError(
                f"Cell {cell['cell_id']!r} has a partial graph cost proxy."
            )
        rows.append(
            {
                "cell_id": str(cell["cell_id"]),
                "regime": str(cell["regime"]),
                "result_json": str(result_json),
                "abs_delta_e": _finite_nonnegative(
                    summary.get("abs_delta_e"), field="abs_delta_e"
                ),
                "expanded_query_work_total": _finite_nonnegative(
                    summary.get("all_branch_query_work_total_diagnostic"),
                    field="all_branch_query_work_total_diagnostic",
                ),
                "expanded_query_work_status": str(
                    summary.get("all_branch_query_work_status_diagnostic")
                ),
                "winning_branch_query_work_total": _finite_nonnegative(
                    summary.get("query_work_total"), field="query_work_total"
                ),
                "winning_branch_query_work_status": str(
                    summary.get("query_work_status")
                ),
                "graph_count_2q_proxy": _finite_nonnegative(
                    graph_proxy.get("count_2q"), field="graph_count_2q_proxy"
                ),
                "graph_depth_proxy": _finite_nonnegative(
                    graph_proxy.get("depth"), field="graph_depth_proxy"
                ),
                "graph_cost_proxy_provenance": {
                    "mode": graph_proxy.get("mode"),
                    "source": graph_proxy.get("source"),
                    "status": graph_proxy.get("status"),
                    "posthoc_fallback_used": bool(graph_proxy_fallback_used),
                    "operator_count": graph_proxy.get("operator_count"),
                    "serialized_term_count": graph_proxy.get(
                        "serialized_term_count"
                    ),
                    "active_pauli_term_count": graph_proxy.get(
                        "active_pauli_term_count"
                    ),
                    "num_qubits": graph_proxy.get("num_qubits"),
                    "coefficient_tolerance": graph_proxy.get(
                        "coefficient_tolerance"
                    ),
                    "terminal_operator_metadata_alignment": graph_proxy.get(
                        "terminal_operator_metadata_alignment"
                    ),
                },
                "controller_round_count": int(
                    summary.get("controller_round_count", 0)
                ),
                "ansatz_depth": int(summary.get("ansatz_depth", 0)),
                "nfev_total": int(summary.get("nfev_total", 0)),
                "selected_batch_cardinality": (
                    dict(summary.get("joint_selector_summary", {})).get(
                        "selected_cardinality"
                    )
                    if isinstance(summary.get("joint_selector_summary"), Mapping)
                    else None
                ),
            }
        )
    status = "complete"
    if failed:
        status = "failed"
    elif missing:
        status = "incomplete"
    if status != "complete":
        return {
            "schema": AGGREGATE_SCHEMA,
            "generated_utc": _utc_now(),
            "status": status,
            "trial_number": int(trial["trial_number"]),
            "policy": policy.as_dict(),
            "missing_cells": missing,
            "failed_cells": failed,
            "completed_cell_count": len(rows),
            "required_cell_count": len(REGIME_ORDER),
        }
    errors = [float(row["abs_delta_e"]) for row in rows]
    objective_values = (
        float(max(errors)),
        float(sum(float(row["expanded_query_work_total"]) for row in rows)),
        float(sum(float(row["graph_count_2q_proxy"]) for row in rows)),
        float(sum(float(row["graph_depth_proxy"]) for row in rows)),
    )
    return {
        "schema": AGGREGATE_SCHEMA,
        "generated_utc": _utc_now(),
        "status": "complete",
        "trial_number": int(trial["trial_number"]),
        "policy": policy.as_dict(),
        "policy_sha256": trial.get("policy_sha256"),
        "regime_count": len(rows),
        "regime_order": list(REGIME_ORDER),
        "objectives": {
            name: float(value)
            for name, value in zip(OBJECTIVE_NAMES, objective_values, strict=True)
        },
        "objective_values": list(objective_values),
        "secondary_diagnostics": {
            "arithmetic_mean_abs_delta_e": float(statistics.fmean(errors)),
            "best_regime_abs_delta_e": float(min(errors)),
            "total_winning_branch_query_work": float(
                sum(
                    float(row["winning_branch_query_work_total"])
                    for row in rows
                )
            ),
            "total_nfev": int(sum(int(row["nfev_total"]) for row in rows)),
            "maximum_ansatz_depth": int(
                max(int(row["ansatz_depth"]) for row in rows)
            ),
        },
        "qiskit_compiled_resources": {
            "status": "post_run_finalists_pending",
            "part_of_optuna_screening_objectives": False,
        },
        "query_axis_contract": "all_expanded_scored_branches_v1",
        "graph_proxy_contract": (
            "proxy_logical_ladder_span_v1_from_terminal_selected_"
            "generator_metadata_v1"
        ),
        "regime_results": rows,
    }


def _front_snapshot(study: Any, *, aggregate_root: Path) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    for trial in study.best_trials:
        aggregate_path = aggregate_root / f"trial_{int(trial.number):06d}.json"
        aggregate = _read_json(aggregate_path) if aggregate_path.is_file() else None
        trials.append(
            {
                "trial_number": int(trial.number),
                "values": [float(value) for value in (trial.values or ())],
                "objectives": {
                    name: float(value)
                    for name, value in zip(
                        OBJECTIVE_NAMES,
                        trial.values or (),
                        strict=True,
                    )
                },
                "policy": trial.user_attrs.get("policy"),
                "aggregate_json": (
                    str(aggregate_path) if aggregate_path.is_file() else None
                ),
                "secondary_diagnostics": (
                    aggregate.get("secondary_diagnostics")
                    if isinstance(aggregate, Mapping)
                    else None
                ),
            }
        )
    return {
        "schema": FRONT_SCHEMA,
        "generated_utc": _utc_now(),
        "status": "screening_front_not_yet_qiskit_compiled",
        "objective_names": list(OBJECTIVE_NAMES),
        "objective_directions": list(OBJECTIVE_DIRECTIONS),
        "trial_count": len(study.trials),
        "front_size": len(trials),
        "trials": trials,
    }


def aggregate_wave(
    *,
    database: Path,
    study_name: str,
    wave_manifest: Path,
    result_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    wave = _read_json(wave_manifest)
    _validate_manifest_hash(wave, source=wave_manifest)
    if str(wave.get("schema")) != WAVE_SCHEMA:
        raise ValueError(f"Unexpected wave schema: {wave.get('schema')!r}.")
    if str(wave.get("study_name")) != str(study_name):
        raise ValueError("Wave and requested Optuna study names differ.")
    optuna = _import_optuna()
    study = optuna.load_study(
        study_name=str(study_name),
        storage=_storage_url(database),
    )
    if str(study.user_attrs.get("scientific_contract_hash")) != str(
        wave.get("scientific_contract_hash")
    ):
        raise ValueError("Wave scientific contract does not match the Optuna study.")
    wave_root = wave_manifest.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_states = {int(trial.number): trial.state for trial in study.trials}
    aggregates: list[dict[str, Any]] = []
    told_complete = 0
    told_failed = 0
    incomplete = 0
    for trial_ref in wave.get("trials", []):
        if not isinstance(trial_ref, Mapping):
            continue
        trial_number = int(trial_ref["trial_number"])
        trial_manifest_path = wave_root / str(trial_ref["trial_manifest"])
        aggregate = aggregate_trial_results(
            trial_manifest=trial_manifest_path,
            wave_root=wave_root,
            result_root=result_root,
        )
        aggregate_path = output_dir / f"trial_{trial_number:06d}.json"
        _write_json(aggregate_path, aggregate)
        aggregates.append(
            {
                "trial_number": trial_number,
                "status": aggregate["status"],
                "aggregate_json": str(aggregate_path),
            }
        )
        current_state = trial_states.get(trial_number)
        if current_state != optuna.trial.TrialState.RUNNING:
            continue
        if str(aggregate["status"]) == "complete":
            study.tell(trial_number, list(aggregate["objective_values"]))
            told_complete += 1
        elif str(aggregate["status"]) == "failed":
            study.tell(trial_number, state=optuna.trial.TrialState.FAIL)
            told_failed += 1
        else:
            incomplete += 1
    front = _front_snapshot(study, aggregate_root=output_dir)
    _write_json(output_dir / "screening_pareto_front.json", front)
    summary = {
        "schema": "paper_i_hh_route_a_optuna_wave_aggregation_v1",
        "generated_utc": _utc_now(),
        "status": "complete" if incomplete == 0 else "incomplete",
        "study_name": str(study_name),
        "wave_manifest": str(wave_manifest),
        "result_root": str(result_root),
        "told_complete": int(told_complete),
        "told_failed": int(told_failed),
        "incomplete": int(incomplete),
        "trial_aggregates": aggregates,
        "screening_front_json": str(output_dir / "screening_pareto_front.json"),
        "qiskit_finalist_compilation_status": "pending",
    }
    _write_json(output_dir / "aggregation_summary.json", summary)
    return summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must be inside the active repository: {path}") from exc


def _runtime_provenance_inputs(wave: Mapping[str, Any]) -> list[dict[str, str]]:
    """Resolve contract source records that must exist inside each worker sandbox."""

    scientific_contract = wave.get("scientific_contract")
    if not isinstance(scientific_contract, Mapping):
        return []
    source_records = scientific_contract.get("source_records")
    if source_records is None:
        return []
    if not isinstance(source_records, Mapping):
        raise ValueError("scientific_contract.source_records must be a mapping.")
    resolved_records: list[dict[str, str]] = []
    seen_paths: dict[str, str] = {}
    for role, raw_record in sorted(source_records.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"Scientific source record {role!r} must be a mapping.")
        raw_path = str(raw_record.get("path") or "").strip()
        expected_sha256 = str(raw_record.get("sha256") or "").strip()
        if not raw_path or not expected_sha256:
            raise ValueError(
                f"Scientific source record {role!r} requires path and sha256."
            )
        relative_path = _repo_relative(REPO_ROOT / raw_path)
        source = REPO_ROOT / relative_path
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing runtime provenance input for {role!r}: {source}"
            )
        actual_sha256 = _sha256_file(source)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Runtime provenance hash mismatch for {role!r}: "
                f"expected {expected_sha256}, found {actual_sha256}."
            )
        relative_text = str(relative_path)
        prior_hash = seen_paths.get(relative_text)
        if prior_hash is not None and prior_hash != expected_sha256:
            raise ValueError(
                f"Conflicting runtime provenance hashes for {relative_text}."
            )
        seen_paths[relative_text] = expected_sha256
        resolved_records.append(
            {
                "role": str(role),
                "path": relative_text,
                "sha256": expected_sha256,
            }
        )
    return resolved_records


def _submit_transfer_inputs(submit_text: str) -> tuple[str, ...]:
    for raw_line in submit_text.splitlines():
        key, separator, value = raw_line.partition("=")
        if separator and key.strip().lower() == "transfer_input_files":
            return tuple(item.strip() for item in value.split(",") if item.strip())
    return ()


def _worker_input_is_transferred(
    relative_path: str | Path,
    *,
    transfer_inputs: Sequence[str],
) -> bool:
    target = Path(str(relative_path))
    for raw_input in transfer_inputs:
        if "$(" in raw_input:
            continue
        candidate = Path(raw_input)
        if candidate == Path("."):
            return True
        try:
            target.relative_to(candidate)
        except ValueError:
            continue
        return True
    return False


def _chtc_resources_for_regime(regime: str) -> tuple[int, int]:
    values = pareto_runner.REGIMES[str(regime)]
    n_ph_work = int(values["n_ph_work"])
    u_value = float(values["u"])
    if n_ph_work >= 4 or u_value >= 8.0:
        return 32_768, 61_440
    return 24_576, 40_960


def _chtc_submit_text(
    *,
    queue_path: Path,
    input_dir: Path,
    batch_output_root: Path,
    batch_name: str,
    runtime_provenance_inputs: Sequence[Mapping[str, str]] = (),
) -> str:
    transfer_input_items = [
        "pipelines",
        "src",
        "docs",
        str(CHTC_WRAPPER),
        "chtc/phase3_optuna/image.sif",
        str(input_dir),
    ]
    for record in runtime_provenance_inputs:
        path = str(record["path"])
        if path not in transfer_input_items:
            transfer_input_items.append(path)
    transfer_inputs = ", ".join(transfer_input_items)
    return "\n".join(
        (
            "universe = vanilla",
            f"executable = {CHTC_WRAPPER}",
            (
                "arguments = $(cell_manifest) "
                f"{batch_output_root} $(output_relative_dir)"
            ),
            "should_transfer_files = YES",
            "when_to_transfer_output = ON_EXIT_OR_EVICT",
            "transfer_executable = True",
            "preserve_relative_paths = True",
            f"transfer_input_files = {transfer_inputs}",
            (
                "transfer_output_files = "
                f"{batch_output_root}/$(output_relative_dir)"
            ),
            "stream_output = False",
            "stream_error = False",
            f"log = logs/{batch_name}.$(Cluster).$(Process).log",
            f"output = logs/{batch_name}.$(Cluster).$(Process).out",
            f"error = logs/{batch_name}.$(Cluster).$(Process).err",
            "requirements = TARGET.HasSIF",
            "request_cpus = 4",
            "request_memory = $(memory_mb)MB",
            "request_disk = $(disk_mb)MB",
            "+MaxRuntime = 259200",
            f'+JobBatchName = "{batch_name}"',
            "notification = Never",
            (
                "queue cell_id, cell_manifest, output_relative_dir, "
                f"memory_mb, disk_mb from {queue_path}"
            ),
            "",
        )
    )


def export_chtc_wave(
    *,
    wave_manifest: Path,
    output_dir: Path,
    batch_output_root: Path,
    batch_name: str,
) -> dict[str, Any]:
    wave = _read_json(wave_manifest)
    _validate_manifest_hash(wave, source=wave_manifest)
    if str(wave.get("schema")) != WAVE_SCHEMA:
        raise ValueError(f"Unexpected wave schema: {wave.get('schema')!r}.")
    if not str(batch_name).strip():
        raise ValueError("batch_name must not be empty.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"CHTC export directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    relative_input_dir = _repo_relative(output_dir)
    if batch_output_root.is_absolute():
        raise ValueError("batch_output_root must be repository-relative.")
    runtime_provenance_inputs = _runtime_provenance_inputs(wave)
    wave_root = wave_manifest.parent
    copied_wave = output_dir / "wave_manifest.json"
    shutil.copy2(wave_manifest, copied_wave)
    copied_settings_diff: str | None = None
    if wave.get("settings_diff") is not None:
        source_diff = wave_root / str(wave["settings_diff"])
        expected_diff_hash = str(wave.get("settings_diff_sha256") or "")
        if not source_diff.is_file() or _sha256_file(source_diff) != expected_diff_hash:
            raise ValueError("Wave settings diff is missing or has drifted.")
        target_diff = output_dir / source_diff.name
        shutil.copy2(source_diff, target_diff)
        copied_settings_diff = str(_repo_relative(target_diff))
    rows: list[dict[str, Any]] = []
    copied_trials: list[str] = []
    copied_cells: list[str] = []
    for trial_ref in wave.get("trials", []):
        if not isinstance(trial_ref, Mapping):
            continue
        source_trial = wave_root / str(trial_ref["trial_manifest"])
        target_trial = output_dir / "trial_manifests" / source_trial.name
        target_trial.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_trial, target_trial)
        copied_trials.append(str(_repo_relative(target_trial)))
        trial_payload = _read_json(source_trial)
        _validate_manifest_hash(trial_payload, source=source_trial)
        for cell_ref in trial_payload.get("cells", []):
            if not isinstance(cell_ref, Mapping):
                continue
            source_cell = wave_root / str(cell_ref["manifest"])
            target_cell = output_dir / "cell_manifests" / source_cell.name
            target_cell.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_cell, target_cell)
            cell = _read_json(target_cell)
            _validate_manifest_hash(cell, source=target_cell)
            if str(cell.get("scientific_contract_hash")) != str(
                wave.get("scientific_contract_hash")
            ):
                raise ValueError(
                    f"Cell contract mismatch during CHTC export: {target_cell}"
                )
            memory_mb, disk_mb = _chtc_resources_for_regime(str(cell["regime"]))
            rows.append(
                {
                    "cell_id": str(cell["cell_id"]),
                    "cell_manifest": str(_repo_relative(target_cell)),
                    "output_relative_dir": str(cell["output_relative_dir"]),
                    "memory_mb": int(memory_mb),
                    "disk_mb": int(disk_mb),
                }
            )
            copied_cells.append(str(_repo_relative(target_cell)))
    if len(rows) != int(wave.get("cell_count", -1)):
        raise ValueError(
            f"Expected {wave.get('cell_count')} CHTC rows, built {len(rows)}."
        )
    queue_path = output_dir / "queue.tsv"
    queue_path.write_text(
        "".join(
            "\t".join(
                (
                    row["cell_id"],
                    row["cell_manifest"],
                    row["output_relative_dir"],
                    str(row["memory_mb"]),
                    str(row["disk_mb"]),
                )
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    submit_path = output_dir / "submit.sub"
    submit_path.write_text(
        _chtc_submit_text(
            queue_path=_repo_relative(queue_path),
            input_dir=relative_input_dir,
            batch_output_root=batch_output_root,
            batch_name=str(batch_name),
            runtime_provenance_inputs=runtime_provenance_inputs,
        ),
        encoding="utf-8",
    )
    source_hashes: dict[str, str] = {}
    for relative in CHTC_SOURCE_HASH_PATHS:
        source = REPO_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing CHTC source-lock file: {source}")
        source_hashes[str(relative)] = _sha256_file(source)
    export = _manifest_with_hash(
        {
            "schema": CHTC_EXPORT_SCHEMA,
            "generated_utc": _utc_now(),
            "run_class": str(wave.get("run_class", "diagnostic_pareto_search")),
            "execution_profile": str(
                wave.get("execution_profile", OPTUNA_EXECUTION_PROFILE)
            ),
            "batch_name": str(batch_name),
            "batch_output_root": str(batch_output_root),
            "wave_manifest": str(_repo_relative(copied_wave)),
            "wave_manifest_sha256": _sha256_file(copied_wave),
            "scientific_contract_hash": wave.get("scientific_contract_hash"),
            "trial_count": int(wave["trial_count"]),
            "cell_count": len(rows),
            "queue_tsv": str(_repo_relative(queue_path)),
            "queue_sha256": _sha256_file(queue_path),
            "submit_file": str(_repo_relative(submit_path)),
            "submit_sha256": _sha256_file(submit_path),
            "worker_optuna_storage_access": False,
            "requested_cpus_per_cell": 4,
            "max_runtime_s": 259_200,
            "source_hashes": source_hashes,
            "runtime_provenance_inputs": runtime_provenance_inputs,
            "settings_diff": copied_settings_diff,
            "settings_diff_sha256": (
                _sha256_file(REPO_ROOT / copied_settings_diff)
                if copied_settings_diff is not None
                else None
            ),
            "trial_manifests": copied_trials,
            "cell_manifests": copied_cells,
            "resource_policy": {
                "nph2_non_u8": {"memory_mb": 24_576, "disk_mb": 40_960},
                "nph4_or_u8": {"memory_mb": 32_768, "disk_mb": 61_440},
            },
        }
    )
    export_path = output_dir / "export_manifest.json"
    _write_json(export_path, export)
    return export


def validate_chtc_export(export_manifest: Path) -> dict[str, Any]:
    export = _read_json(export_manifest)
    _validate_manifest_hash(export, source=export_manifest)
    if str(export.get("schema")) != CHTC_EXPORT_SCHEMA:
        raise ValueError(f"Unexpected CHTC export schema: {export.get('schema')!r}.")
    wave_path = REPO_ROOT / str(export["wave_manifest"])
    wave = _read_json(wave_path)
    _validate_manifest_hash(wave, source=wave_path)
    expected_runtime_provenance = _runtime_provenance_inputs(wave)
    declared_runtime_provenance = export.get("runtime_provenance_inputs", [])
    if declared_runtime_provenance != expected_runtime_provenance:
        raise ValueError(
            "CHTC export runtime provenance manifest does not match the "
            "scientific contract."
        )
    checks: list[dict[str, Any]] = []
    for field, hash_field in (
        ("wave_manifest", "wave_manifest_sha256"),
        ("queue_tsv", "queue_sha256"),
        ("submit_file", "submit_sha256"),
    ):
        path = REPO_ROOT / str(export[field])
        actual = _sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "path": str(path),
                "status": "pass" if actual == export[hash_field] else "fail",
                "expected_sha256": export[hash_field],
                "actual_sha256": actual,
            }
        )
    if export.get("settings_diff") is not None:
        diff_path = REPO_ROOT / str(export["settings_diff"])
        actual_diff_hash = _sha256_file(diff_path) if diff_path.is_file() else None
        checks.append(
            {
                "path": str(diff_path),
                "status": (
                    "pass"
                    if actual_diff_hash == export.get("settings_diff_sha256")
                    else "fail"
                ),
                "expected_sha256": export.get("settings_diff_sha256"),
                "actual_sha256": actual_diff_hash,
            }
        )
    for relative, expected in dict(export.get("source_hashes", {})).items():
        path = REPO_ROOT / str(relative)
        actual = _sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "path": str(path),
                "status": "pass" if actual == expected else "fail",
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
    submit_path = REPO_ROOT / str(export["submit_file"])
    submit_text = submit_path.read_text(encoding="utf-8") if submit_path.is_file() else ""
    transfer_inputs = _submit_transfer_inputs(submit_text)
    for record in expected_runtime_provenance:
        if not isinstance(record, Mapping):
            raise ValueError("runtime_provenance_inputs entries must be mappings.")
        relative = str(record.get("path") or "")
        expected = str(record.get("sha256") or "")
        path = REPO_ROOT / relative
        actual = _sha256_file(path) if path.is_file() else None
        worker_visible = _worker_input_is_transferred(
            relative,
            transfer_inputs=transfer_inputs,
        )
        checks.append(
            {
                "path": str(path),
                "check": "runtime_provenance_input",
                "status": (
                    "pass"
                    if actual == expected and worker_visible
                    else "fail"
                ),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "worker_visible": worker_visible,
            }
        )
    cell_paths = [REPO_ROOT / str(path) for path in export.get("cell_manifests", [])]
    for path in cell_paths:
        cell = _read_json(path)
        _validate_manifest_hash(cell, source=path)
        if str(cell.get("scientific_contract_hash")) != str(
            export.get("scientific_contract_hash")
        ):
            raise ValueError(f"CHTC cell has a different scientific contract: {path}")
    queue_path = REPO_ROOT / str(export["queue_tsv"])
    queue_rows = [line for line in queue_path.read_text(encoding="utf-8").splitlines() if line]
    if len(queue_rows) != int(export["cell_count"]):
        raise ValueError(
            f"CHTC queue has {len(queue_rows)} rows, expected {export['cell_count']}."
        )
    failed = [check for check in checks if check["status"] != "pass"]
    if failed:
        raise ValueError("CHTC export hash validation failed: " + json.dumps(failed))
    return {
        "schema": "paper_i_hh_route_a_optuna_chtc_preflight_v1",
        "generated_utc": _utc_now(),
        "status": "pass",
        "batch_name": export["batch_name"],
        "trial_count": int(export["trial_count"]),
        "cell_count": int(export["cell_count"]),
        "hash_check_count": len(checks),
        "worker_optuna_storage_access": False,
        "submit_file": export["submit_file"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan-wave",
        help="Ask Optuna for policies and write immutable six-regime cell manifests.",
    )
    plan.add_argument("--database", type=Path, required=True)
    plan.add_argument("--study-name", required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--n-trials", type=int, required=True)
    plan.add_argument("--max-depth", type=int, default=15)
    plan.add_argument("--batch-mode", choices=BATCH_MODES, default="combinatorial")
    plan.add_argument("--sampler-seed", type=int, default=7)
    plan.add_argument("--population-size", type=int, default=16)

    fixed = subparsers.add_parser(
        "plan-wave11-legal",
        help=(
            "Write the immutable six-regime Wave-11 policy with cutoff-generic "
            "exact projected/grouped legal children."
        ),
    )
    fixed.add_argument("--output-dir", type=Path, required=True)
    fixed.add_argument("--max-depth", type=int, default=15)

    adaptive_whitened = subparsers.add_parser(
        "plan-wave11-adaptive-whitened-r30",
        help=(
            "Write the fresh six-regime R30 L25 policy with supported-metric "
            "whitening, adaptive-unbounded trust radius, and exhaustion retry."
        ),
    )
    adaptive_whitened.add_argument("--output-dir", type=Path, required=True)
    adaptive_whitened.add_argument("--max-depth", type=int, default=30)

    jr_l10_r50 = subparsers.add_parser(
        "plan-jr-l10-rollback-free-r50",
        help=(
            "Write the fresh six-regime rollback-free JR-L10 R50 matrix with "
            "supported-metric whitening and exact guarded warm starts."
        ),
    )
    jr_l10_r50.add_argument("--output-dir", type=Path, required=True)
    jr_l10_r50.add_argument("--max-depth", type=int, default=50)

    cell = subparsers.add_parser(
        "run-cell",
        help="Run or dry-run one immutable regime cell without touching Optuna storage.",
    )
    cell.add_argument("--cell-manifest", type=Path, required=True)
    cell.add_argument("--output-root", type=Path, required=True)
    cell.add_argument("--dry-run", action="store_true")
    cell.add_argument("--gradient-workers", type=int, default=1)
    cell.add_argument("--beam-parent-workers", type=int, default=1)
    cell.add_argument("--runtime-split-child-workers", type=int, default=0)
    cell.add_argument("--joint-pair-workers", type=int, default=1)

    aggregate = subparsers.add_parser(
        "aggregate-wave",
        help="Aggregate completed six-regime trials and tell the local Optuna study.",
    )
    aggregate.add_argument("--database", type=Path, required=True)
    aggregate.add_argument("--study-name", required=True)
    aggregate.add_argument("--wave-manifest", type=Path, required=True)
    aggregate.add_argument("--result-root", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)

    export = subparsers.add_parser(
        "export-chtc",
        help="Export a validated wave as immutable CHTC cell records and submit file.",
    )
    export.add_argument("--wave-manifest", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--batch-output-root", type=Path, required=True)
    export.add_argument("--batch-name", required=True)

    preflight = subparsers.add_parser(
        "preflight-chtc",
        help="Verify CHTC manifests, source hashes, queue rows, and submit file.",
    )
    preflight.add_argument("--export-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan-wave":
        payload = plan_wave(
            database=Path(args.database),
            study_name=str(args.study_name),
            output_dir=Path(args.output_dir),
            n_trials=int(args.n_trials),
            max_depth=int(args.max_depth),
            batch_mode=str(args.batch_mode),
            sampler_seed=int(args.sampler_seed),
            population_size=int(args.population_size),
        )
    elif args.command == "plan-wave11-legal":
        payload = plan_wave11_legal_matrix(
            output_dir=Path(args.output_dir),
            max_depth=int(args.max_depth),
        )
    elif args.command == "plan-wave11-adaptive-whitened-r30":
        payload = plan_wave11_adaptive_whitened_r30_matrix(
            output_dir=Path(args.output_dir),
            max_depth=int(args.max_depth),
        )
    elif args.command == "plan-jr-l10-rollback-free-r50":
        payload = plan_jr_l10_rollback_free_r50_matrix(
            output_dir=Path(args.output_dir),
            max_depth=int(args.max_depth),
        )
    elif args.command == "run-cell":
        payload = run_cell(
            cell_manifest=Path(args.cell_manifest),
            output_root=Path(args.output_root),
            dry_run=bool(args.dry_run),
            gradient_workers=int(args.gradient_workers),
            beam_parent_workers=int(args.beam_parent_workers),
            runtime_split_child_workers=int(args.runtime_split_child_workers),
            joint_pair_workers=int(args.joint_pair_workers),
        )
    elif args.command == "aggregate-wave":
        payload = aggregate_wave(
            database=Path(args.database),
            study_name=str(args.study_name),
            wave_manifest=Path(args.wave_manifest),
            result_root=Path(args.result_root),
            output_dir=Path(args.output_dir),
        )
    elif args.command == "export-chtc":
        payload = export_chtc_wave(
            wave_manifest=Path(args.wave_manifest),
            output_dir=Path(args.output_dir),
            batch_output_root=Path(args.batch_output_root),
            batch_name=str(args.batch_name),
        )
    else:
        payload = validate_chtc_export(Path(args.export_manifest))
    print(json.dumps(_json_ready(payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
