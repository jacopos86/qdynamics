#!/usr/bin/env python3
"""Factorial score ablation for the Paper-III QSE acquisition rule.

The diagnostic crosses four spectral scores with the additive novelty reward
enabled and disabled. The full-frame novelty floor remains enabled in every
cell. Every growth prefix is solved both before and after a separately
certified exchange attempt so exchange cannot conceal acquisition behavior.
Optional profiles screen with the residualized 2x2 score and rerank a declared
shortlist by the full Ky Fan objective. Screening work remains a vector of
counts rather than being folded into an unsupported scalar cost conversion.

Statevector diagnostics only; never feeds controller decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.paper_iii_matched_accuracy_campaign import REGIME_SETS
from pipelines.exact_bench.paper_iii_qse_measurement_cost import (
    _element_words,
    _polynomial_words,
    _qwc_cover,
    _word_product,
)
from pipelines.qse_spectra.core import QSEBasisVectorPolicy, compute_qse_spectra
from pipelines.qse_spectra.exchange_maintenance import (
    QSEExchangeConfig,
    exchange_maintenance_payload,
    run_qse_exchange_maintenance,
)
from pipelines.qse_spectra.record_selection import (
    GEOMETRY_SCORE_FULL_PENCIL_KY_FAN,
    GEOMETRY_SCORE_NORMALIZED_ANGULAR,
    GEOMETRY_SCORE_RAW_OVERLAP_RITZ_GAIN,
    GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN,
    GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST,
    StaticRecordSelectionConfig,
    select_static_qse_records,
    static_record_selection_payload,
)
from pipelines.qse_spectra.paper_iii_problem import PaperIIIProblem, load_problem
from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_score_factorial_20260828_v1/score_factorial.json"
)
_Q0_POLICY = QSEBasisVectorPolicy(
    reference_projection="q0", basis_vector_normalization="raw_projected"
)
_TARGET_ROOTS = 6

FACTORIAL_SCORE_MODES = (
    GEOMETRY_SCORE_NORMALIZED_ANGULAR,
    GEOMETRY_SCORE_RAW_OVERLAP_RITZ_GAIN,
    GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN,
    GEOMETRY_SCORE_FULL_PENCIL_KY_FAN,
)
_METHOD_LOCK_ERROR_TOLERANCES = (1.0e-2, 1.0e-4)
_METHOD_LOCK_MAX_GROWTH_ROUND_LAG = 1
_METHOD_LOCK_ORACLE_PROFILE_ID = (
    f"{GEOMETRY_SCORE_FULL_PENCIL_KY_FAN}__novelty_reward_off"
)


@dataclass(frozen=True)
class ScoreAblationProfile:
    profile_id: str
    score_mode: str
    additive_novelty_reward: bool
    selector_config: StaticRecordSelectionConfig


def factorial_score_profiles(
    *, max_records: int, target_root_count: int
) -> tuple[ScoreAblationProfile, ...]:
    """Return the locked four-by-two selector factorial."""

    profiles: list[ScoreAblationProfile] = []
    for score_mode in FACTORIAL_SCORE_MODES:
        for reward in (True, False):
            reward_label = "on" if reward else "off"
            profiles.append(
                ScoreAblationProfile(
                    profile_id=f"{score_mode}__novelty_reward_{reward_label}",
                    score_mode=str(score_mode),
                    additive_novelty_reward=bool(reward),
                    selector_config=StaticRecordSelectionConfig(
                        mode="geometry_selected",
                        max_records=int(max_records),
                        geometry_target_roots=int(target_root_count),
                        geometry_score_mode=str(score_mode),
                        geometry_additive_novelty_reward=bool(reward),
                        geometry_require_supported_image=True,
                        geometry_min_metric_novelty=1.0e-12,
                        geometry_cost_discount_alpha=1.0,
                    ),
                )
            )
    return tuple(profiles)


def shortlist_score_profiles(
    *,
    shortlist_sizes: Sequence[int],
    max_records: int,
    target_root_count: int,
) -> tuple[ScoreAblationProfile, ...]:
    """Return residualized-screen/full-pencil-rerank profiles."""

    profiles: list[ScoreAblationProfile] = []
    for shortlist_size in shortlist_sizes:
        size = int(shortlist_size)
        if size < 1:
            raise ValueError("shortlist sizes must be positive")
        profiles.append(
            ScoreAblationProfile(
                profile_id=(
                    "residualized_2x2_then_ky_fan"
                    f"__shortlist_{size}"
                ),
                score_mode=GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST,
                additive_novelty_reward=False,
                selector_config=StaticRecordSelectionConfig(
                    mode="geometry_selected",
                    max_records=int(max_records),
                    geometry_target_roots=int(target_root_count),
                    geometry_score_mode=(
                        GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST
                    ),
                    geometry_additive_novelty_reward=False,
                    geometry_require_supported_image=True,
                    geometry_ky_fan_shortlist_size=size,
                    geometry_min_metric_novelty=1.0e-12,
                    geometry_cost_discount_alpha=1.0,
                ),
            )
        )
    return tuple(profiles)


def _pair_measurement_work(
    problem: Any,
    pairs: Sequence[tuple[int, int]],
    *,
    hamiltonian_words: Sequence[str],
    candidate_indices: Sequence[int] = (),
) -> tuple[dict[str, Any], frozenset[str]]:
    """Return exact Pauli-word work for declared independent pencil pairs."""

    overlap_words: set[str] = set()
    projected_hamiltonian_words: set[str] = set()
    naive_terms = 0
    element_word_cache: dict[int, tuple[str, ...]] = {}
    for left, right in pairs:
        left_index = int(left)
        right_index = int(right)
        for index in (left_index, right_index):
            if index not in element_word_cache:
                element_word_cache[index] = _element_words(
                    problem.basis[index], nq=int(problem.num_qubits)
                )
        products = {
            _word_product(left_word, right_word)
            for left_word in element_word_cache[left_index]
            for right_word in element_word_cache[right_index]
        }
        overlap_words |= products
        naive_terms += len(products)
        h_products = {
            _word_product(product, hamiltonian_word)
            for product in products
            for hamiltonian_word in hamiltonian_words
        }
        projected_hamiltonian_words |= h_products
        naive_terms += len(h_products)
    all_words = frozenset(overlap_words | projected_hamiltonian_words)
    candidates = [int(index) for index in candidate_indices]
    payload: dict[str, Any] = {
        "pair_count": len(pairs),
        "naive_terms": int(naive_terms),
        "distinct_words": len(all_words),
        "qwc_groups": _qwc_cover(sorted(all_words), nq=int(problem.num_qubits)),
    }
    if candidates:
        payload = {
            "candidate_original_indices": candidates,
            "candidate_count": len(candidates),
            "candidate_compiled_K_sum": float(
                sum(float(problem.costs[index]) for index in candidates)
            ),
            **payload,
        }
    return payload, all_words


def _candidate_row_pairs(
    retained_indices: Sequence[int], candidate_indices: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    retained = [int(index) for index in retained_indices]
    for candidate in (int(index) for index in candidate_indices):
        pairs.extend((min(index, candidate), max(index, candidate)) for index in retained)
        pairs.append((candidate, candidate))
    return tuple(dict.fromkeys(pairs))


def _selection_acquisition_work(
    problem: Any,
    retained_indices: Sequence[int],
    screening: Mapping[str, Any],
    *,
    hamiltonian_words: Sequence[str],
) -> tuple[dict[str, Any], dict[str, frozenset[str]]]:
    """Account for candidate-row acquisition without double-counting reuse."""

    retained = [int(index) for index in retained_indices]
    two_dimensional = [
        int(index)
        for index in screening.get("two_dimensional_candidate_original_indices", ())
    ]
    reranked = [
        int(index)
        for index in screening.get(
            "full_proposed_pencil_candidate_original_indices", ()
        )
    ]
    retained_pairs = tuple(
        (retained[left], retained[right])
        for left in range(len(retained))
        for right in range(left, len(retained))
    )
    retained_payload, retained_words = _pair_measurement_work(
        problem, retained_pairs, hamiltonian_words=hamiltonian_words
    )
    screen_payload, screen_words = _pair_measurement_work(
        problem,
        _candidate_row_pairs(retained, two_dimensional),
        hamiltonian_words=hamiltonian_words,
        candidate_indices=two_dimensional,
    )
    rerank_payload, rerank_words = _pair_measurement_work(
        problem,
        _candidate_row_pairs(retained, reranked),
        hamiltonian_words=hamiltonian_words,
        candidate_indices=reranked,
    )
    already_acquired = retained_words | screen_words
    rerank_incremental_words = frozenset(rerank_words - already_acquired)
    total_words = frozenset(already_acquired | rerank_words)
    payload = {
        "measurement_model": (
            "candidate rows contain S and H elements against the retained support; "
            "candidate-candidate cross terms are excluded"
        ),
        "cost_vector_policy": (
            "pair counts, Pauli words, QWC groups, and candidate compiled K are "
            "reported separately and are not scalarized"
        ),
        "retained_pencil": retained_payload,
        "residualized_2x2_screen": screen_payload,
        "ky_fan_rerank": {
            "raw_candidate_rows": rerank_payload,
            "incremental_after_current_round_reuse": {
                "distinct_words": len(rerank_incremental_words),
                "qwc_groups": _qwc_cover(
                    sorted(rerank_incremental_words), nq=int(problem.num_qubits)
                ),
            },
        },
        "total_with_current_round_reuse": {
            "distinct_words": len(total_words),
            "qwc_groups": _qwc_cover(
                sorted(total_words), nq=int(problem.num_qubits)
            ),
        },
    }
    return payload, {
        "retained_pencil": retained_words,
        "residualized_2x2_screen": screen_words,
        "ky_fan_rerank_raw": rerank_words,
        "ky_fan_rerank_incremental": rerank_incremental_words,
        "total": total_words,
    }


def _cumulative_acquisition_payload(
    problem: Any,
    words_by_stage: Mapping[str, set[str]],
    event_totals: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    stage_payload: dict[str, Any] = {}
    for stage in (
        "retained_pencil",
        "residualized_2x2_screen",
        "ky_fan_rerank_raw",
        "ky_fan_rerank_incremental",
    ):
        words = words_by_stage[stage]
        stage_payload[stage] = {
            **dict(event_totals[stage]),
            "distinct_words": len(words),
            "qwc_groups": _qwc_cover(sorted(words), nq=int(problem.num_qubits)),
        }
    total_words = words_by_stage["total"]
    return {
        "reuse_policy": "global Pauli-word reuse across candidates and growth rounds",
        **stage_payload,
        "total_with_cross_round_reuse": {
            "distinct_words": len(total_words),
            "qwc_groups": _qwc_cover(
                sorted(total_words), nq=int(problem.num_qubits)
            ),
        },
    }


def _first_pre_exchange_crossing(
    profile: Mapping[str, Any], *, error_tolerance: float
) -> int | None:
    for row in profile.get("trace", ()):
        reading = row.get("pre_exchange") or {}
        error = reading.get("max_root_abs_error")
        if (
            reading.get("target_window_status") == "TARGET_WINDOW_RESOLVED"
            and error is not None
            and float(error) <= float(error_tolerance)
        ):
            return int(row["growth_round"])
    return None


def _method_lock_assessment(
    cases: Sequence[Mapping[str, Any]],
    *,
    target_root_count: int,
    error_tolerances: Sequence[float] = _METHOD_LOCK_ERROR_TOLERANCES,
    max_growth_round_lag: int = _METHOD_LOCK_MAX_GROWTH_ROUND_LAG,
) -> dict[str, Any]:
    shortlist_sizes = (int(target_root_count), 2 * int(target_root_count))
    candidate_profile_ids = tuple(
        f"{GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST}__shortlist_{size}"
        for size in shortlist_sizes
    )
    candidates: dict[str, Any] = {}
    for profile_id, shortlist_size in zip(candidate_profile_ids, shortlist_sizes):
        comparisons: list[dict[str, Any]] = []
        candidate_qualified = True
        for case in cases:
            regime = str(case["problem"]["regime"])
            oracle = case["profiles"][_METHOD_LOCK_ORACLE_PROFILE_ID]
            candidate = case["shortlist_profiles"][profile_id]
            for tolerance in error_tolerances:
                oracle_round = _first_pre_exchange_crossing(
                    oracle, error_tolerance=float(tolerance)
                )
                candidate_round = _first_pre_exchange_crossing(
                    candidate, error_tolerance=float(tolerance)
                )
                qualified = bool(
                    oracle_round is not None
                    and candidate_round is not None
                    and candidate_round <= oracle_round + int(max_growth_round_lag)
                )
                if oracle_round is None:
                    reason = "ORACLE_HORIZON_INSUFFICIENT"
                elif candidate_round is None:
                    reason = "CANDIDATE_DID_NOT_CROSS"
                elif not qualified:
                    reason = "GROWTH_ROUND_LAG_EXCEEDED"
                else:
                    reason = "WITHIN_LOCK_TOLERANCE"
                comparisons.append(
                    {
                        "regime": regime,
                        "error_tolerance": float(tolerance),
                        "evidence_stage": "pre_exchange",
                        "oracle_growth_round": oracle_round,
                        "candidate_growth_round": candidate_round,
                        "growth_round_lag": (
                            int(candidate_round - oracle_round)
                            if oracle_round is not None and candidate_round is not None
                            else None
                        ),
                        "qualified": bool(qualified),
                        "reason": reason,
                    }
                )
                candidate_qualified = candidate_qualified and qualified
        candidates[profile_id] = {
            "shortlist_size": int(shortlist_size),
            "qualified": bool(candidate_qualified),
            "comparisons": comparisons,
        }

    recommended_profile_id = next(
        (
            profile_id
            for profile_id in candidate_profile_ids
            if bool(candidates[profile_id]["qualified"])
        ),
        None,
    )
    no_candidate_row_acquisition_advantage = bool(recommended_profile_id)
    if recommended_profile_id is not None:
        for case in cases:
            shortlist_profile = case["shortlist_profiles"][recommended_profile_id]
            oracle_profile = case["profiles"][_METHOD_LOCK_ORACLE_PROFILE_ID]
            for row in shortlist_profile.get("trace", ()):
                screening = (row.get("selector_round") or {}).get("screening") or {}
                if set(screening.get("two_dimensional_candidate_original_indices", ())) != set(
                    screening.get("admissible_candidate_original_indices", ())
                ):
                    no_candidate_row_acquisition_advantage = False
                incremental = (row.get("selection_acquisition_work") or {}).get(
                    "ky_fan_rerank", {}
                ).get("incremental_after_current_round_reuse", {})
                if int(incremental.get("distinct_words", 0)) != 0:
                    no_candidate_row_acquisition_advantage = False
            for row in oracle_profile.get("trace", ()):
                screening = (row.get("selector_round") or {}).get("screening") or {}
                if set(
                    screening.get(
                        "full_proposed_pencil_candidate_original_indices", ()
                    )
                ) != set(screening.get("admissible_candidate_original_indices", ())):
                    no_candidate_row_acquisition_advantage = False

    if recommended_profile_id is None:
        method_identity_recommendation = {
            "status": "FULL_PENCIL_KY_FAN_PREFERRED",
            "profile_id": _METHOD_LOCK_ORACLE_PROFILE_ID,
            "reason": "NO_SHORTLIST_PROFILE_QUALIFIED",
            "shortlist_profile_id": None,
        }
    elif no_candidate_row_acquisition_advantage:
        method_identity_recommendation = {
            "status": "FULL_PENCIL_KY_FAN_PREFERRED",
            "profile_id": _METHOD_LOCK_ORACLE_PROFILE_ID,
            "reason": "SHORTLIST_HAS_NO_CANDIDATE_ROW_ACQUISITION_ADVANTAGE",
            "shortlist_profile_id": recommended_profile_id,
        }
    else:
        method_identity_recommendation = {
            "status": "SHORTLIST_PROFILE_PREFERRED",
            "profile_id": recommended_profile_id,
            "reason": "QUALIFIED_WITH_CANDIDATE_ROW_ACQUISITION_ADVANTAGE",
            "shortlist_profile_id": recommended_profile_id,
        }
    return {
        "schema_version": "paper_iii_qse_method_lock_v1",
        "policy": (
            "prefer m=R, otherwise m=2R, only when every nph1 regime crosses "
            "the declared pre-exchange accuracy thresholds within one growth "
            "round of the no-novelty full-pencil Ky Fan oracle"
        ),
        "oracle_profile_id": _METHOD_LOCK_ORACLE_PROFILE_ID,
        "error_tolerances": [float(value) for value in error_tolerances],
        "max_growth_round_lag": int(max_growth_round_lag),
        "candidates": candidates,
        "recommendation": {
            "status": (
                "SHORTLIST_PROFILE_QUALIFIED"
                if recommended_profile_id is not None
                else "NO_SHORTLIST_PROFILE_QUALIFIED"
            ),
            "profile_id": recommended_profile_id,
            "shortlist_size": (
                int(candidates[recommended_profile_id]["shortlist_size"])
                if recommended_profile_id is not None
                else None
            ),
            "rule": "prefer_m_equals_R_then_m_equals_2R",
        },
        "method_identity_recommendation": method_identity_recommendation,
    }


def _ritz_states(result: Any, *, target_root_count: int) -> np.ndarray:
    vectors = tuple(result.matrices.basis_matrix_vectors)
    if not vectors:
        return np.empty((int(result.matrices.hilbert_dim), 0), dtype=complex)
    basis_matrix = np.column_stack(
        [np.asarray(vector, dtype=complex).reshape(-1) for vector in vectors]
    )
    count = min(int(target_root_count), int(result.eigenvectors_basis.shape[1]))
    states = basis_matrix @ np.asarray(result.eigenvectors_basis[:, :count], dtype=complex)
    for column in range(states.shape[1]):
        norm = float(np.linalg.norm(states[:, column]))
        if norm > 0.0:
            states[:, column] /= norm
    return states


def _state_matching(previous: np.ndarray, current: np.ndarray) -> dict[str, Any] | None:
    if previous.shape[1] == 0 or current.shape[1] == 0:
        return None
    amplitudes = previous.conj().T @ current
    overlap_squared = np.abs(amplitudes) ** 2
    previous_rows, current_cols = linear_sum_assignment(-overlap_squared)
    singular_values = np.linalg.svd(amplitudes, compute_uv=False)
    return {
        "policy": "maximum_metric_overlap_assignment",
        "assignments": [
            {
                "previous_root": int(row),
                "current_root": int(col),
                "overlap_squared": float(overlap_squared[row, col]),
            }
            for row, col in zip(previous_rows, current_cols)
        ],
        "principal_cosines": [float(value) for value in singular_values],
        "minimum_principal_cosine": (
            float(np.min(singular_values)) if singular_values.size else None
        ),
    }


def _support_reading(
    problem: Any,
    indices: Sequence[int],
    *,
    target_root_count: int,
    compiled_hamiltonian: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    support = [int(index) for index in indices]
    names = [str(problem.basis[index].name) for index in support]
    references = [float(value) for value in problem.references[: int(target_root_count)]]
    base: dict[str, Any] = {
        "selected_support": {"indices": support, "names": names},
        "support_size": len(support),
        **problem.resource_triple(support),
        "estimator_cost": dict(problem.estimator_cost(support)),
    }
    try:
        result = compute_qse_spectra(
            problem.hamiltonian,
            problem.ground,
            tuple(problem.basis[index] for index in support),
            basis_vector_policy=_Q0_POLICY,
        )
    except ValueError as exc:
        return {
            **base,
            "solve_status": "SOLVE_FAILED",
            "solve_error": str(exc),
            "target_window_status": "TARGET_WINDOW_NOT_RESOLVED",
            "retained_rank": 0,
            "root_energies": [],
            "root_abs_errors": [None for _ in references],
            "available_max_root_abs_error": None,
            "max_root_abs_error": None,
            "ky_fan_sum": None,
            "ky_fan_abs_error": None,
            "target_root_energy": None,
            "gap_to_next_root": None,
            "generalized_residual_norms": [],
            "physical_residual_norms": [],
            "overlap_condition_estimate": None,
            "matching_to_previous": None,
        }, np.empty((len(problem.ground), 0), dtype=complex)

    energies = np.asarray(result.eigenvalues, dtype=float).reshape(-1)
    target_count = int(target_root_count)
    target_energies = [float(value) for value in energies[:target_count]]
    errors = [
        abs(float(energies[root]) - references[root]) if root < energies.size else None
        for root in range(len(references))
    ]
    available_errors = [float(value) for value in errors if value is not None]
    target_window_resolved = energies.size >= target_count
    states = _ritz_states(result, target_root_count=target_count)
    physical_residuals: list[float] = []
    for root in range(states.shape[1]):
        state = states[:, root]
        residual = apply_compiled_polynomial(state, compiled_hamiltonian) - float(
            energies[root]
        ) * state
        physical_residuals.append(float(np.linalg.norm(residual)))
    ky_fan_sum = (
        float(np.sum(energies[:target_count])) if energies.size >= target_count else None
    )
    reference_ky_fan = float(sum(references))
    target_root_energy = (
        float(energies[target_count - 1]) if energies.size >= target_count else None
    )
    gap_to_next = (
        float(energies[target_count] - energies[target_count - 1])
        if energies.size > target_count
        else None
    )
    condition = result.overlap_condition_estimate
    return {
        **base,
        "solve_status": "SOLVED",
        "target_window_status": (
            "TARGET_WINDOW_RESOLVED"
            if target_window_resolved
            else "TARGET_WINDOW_NOT_RESOLVED"
        ),
        "retained_rank": int(result.retained_rank),
        "root_energies": target_energies,
        "root_abs_errors": errors,
        "available_max_root_abs_error": (
            max(available_errors) if available_errors else None
        ),
        "max_root_abs_error": (
            max(available_errors) if target_window_resolved and available_errors else None
        ),
        "ky_fan_sum": ky_fan_sum,
        "ky_fan_abs_error": (
            abs(float(ky_fan_sum) - reference_ky_fan) if ky_fan_sum is not None else None
        ),
        "target_root_energy": target_root_energy,
        "gap_to_next_root": gap_to_next,
        "generalized_residual_norms": [
            float(value) for value in result.generalized_residual_norms[:target_count]
        ],
        "physical_residual_norms": physical_residuals,
        "overlap_condition_estimate": float(condition) if condition is not None else None,
        "matching_to_previous": None,
    }, states


def run_factorial_score_ablation(
    problem: Any,
    *,
    k_max: int,
    target_root_count: int = _TARGET_ROOTS,
    exchange_max_rounds: int = 4,
    shortlist_sizes: Sequence[int] = (),
    profile_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the eight-cell factorial plus optional shortlist profiles."""

    cap = min(int(k_max), len(problem.basis))
    if cap < 1:
        raise ValueError("k_max must admit at least one growth round.")
    if int(target_root_count) < 1:
        raise ValueError("target_root_count must be positive.")
    compiled_hamiltonian = compile_polynomial_action(problem.hamiltonian)
    hamiltonian_words = _polynomial_words(problem.hamiltonian)
    profiles_payload: dict[str, Any] = {}
    shortlist_payload: dict[str, Any] = {}
    all_factorial_profiles = factorial_score_profiles(
        max_records=len(problem.basis), target_root_count=int(target_root_count)
    )
    all_shortlist_profiles = shortlist_score_profiles(
        shortlist_sizes=shortlist_sizes,
        max_records=len(problem.basis),
        target_root_count=int(target_root_count),
    )
    all_profiles = (*all_factorial_profiles, *all_shortlist_profiles)
    profiles_by_id = {profile.profile_id: profile for profile in all_profiles}
    if profile_ids is None:
        selected_profiles = all_profiles
    else:
        requested_profile_ids = tuple(str(value) for value in profile_ids)
        if not requested_profile_ids:
            raise ValueError("profile_ids must contain at least one profile when supplied.")
        if len(set(requested_profile_ids)) != len(requested_profile_ids):
            raise ValueError("profile_ids must not contain duplicates.")
        unknown = [
            profile_id
            for profile_id in requested_profile_ids
            if profile_id not in profiles_by_id
        ]
        if unknown:
            raise ValueError(f"Unknown score-ablation profile_ids: {unknown!r}.")
        selected_profiles = tuple(
            profiles_by_id[profile_id] for profile_id in requested_profile_ids
        )
    shortlist_profile_ids = {
        profile.profile_id for profile in all_shortlist_profiles
    }
    for profile in selected_profiles:
        selection = select_static_qse_records(
            problem.basis,
            config=profile.selector_config,
            hamiltonian=problem.hamiltonian,
            prepared_state=problem.ground,
            basis_vector_policy=_Q0_POLICY,
            compiled_costs=problem.costs,
        )
        order = [int(index) for index in selection.selected_original_indices]
        selector_rounds = list((selection.geometry_stop or {}).get("round_trace") or ())
        decisions = {
            int(row["original_basis_index"]): row for row in selection.candidate_decisions
        }
        trace: list[dict[str, Any]] = []
        screening_count_fields = (
            "admissible_candidate_count",
            "angular_residual_overlap_evaluations",
            "two_dimensional_pencil_evaluations",
            "full_proposed_pencil_evaluations",
            "full_proposed_hamiltonian_row_elements",
        )
        screening_totals = {
            "selection_round_count": 0,
            **{field: 0 for field in screening_count_fields},
        }
        acquisition_words_by_stage: dict[str, set[str]] = {
            stage: set()
            for stage in (
                "retained_pencil",
                "residualized_2x2_screen",
                "ky_fan_rerank_raw",
                "ky_fan_rerank_incremental",
                "total",
            )
        }
        acquisition_event_totals: dict[str, dict[str, float | int]] = {
            stage: {
                "pair_events": 0,
                "naive_term_events": 0,
                "candidate_compiled_K_event_sum": 0.0,
            }
            for stage in (
                "retained_pencil",
                "residualized_2x2_screen",
                "ky_fan_rerank_raw",
                "ky_fan_rerank_incremental",
            )
        }
        previous_pre_states = np.empty((len(problem.ground), 0), dtype=complex)
        for growth_round, selected_index in enumerate(order[:cap], start=1):
            pre_indices = order[:growth_round]
            pre, pre_states = _support_reading(
                problem,
                pre_indices,
                target_root_count=int(target_root_count),
                compiled_hamiltonian=compiled_hamiltonian,
            )
            pre["matching_to_previous"] = _state_matching(previous_pre_states, pre_states)
            previous_pre_states = pre_states
            try:
                exchange_result = run_qse_exchange_maintenance(
                    problem.basis,
                    pre_indices,
                    problem.costs,
                    hamiltonian=problem.hamiltonian,
                    prepared_state=problem.ground,
                    basis_vector_policy=_Q0_POLICY,
                    config=QSEExchangeConfig(
                        max_rounds=int(exchange_max_rounds),
                        target_root_count=int(target_root_count),
                        insertion_shortlist_size=min(8, len(problem.basis)),
                    ),
                )
                post_indices = [int(index) for index in exchange_result.final_indices]
                exchange_full = exchange_maintenance_payload(exchange_result)
                exchange = {
                    "status": "COMPLETED",
                    "committed_patch_count": int(exchange_full["committed_patch_count"]),
                    "initial": dict(exchange_full["initial"]),
                    "final": dict(exchange_full["final"]),
                    "round_count": len(exchange_result.rounds),
                    "committed_patches": [
                        dict(row["committed_patch"])
                        for row in exchange_result.rounds
                        if row["committed_patch"] is not None
                    ],
                }
            except ValueError as exc:
                post_indices = list(pre_indices)
                exchange = {
                    "status": "EXCHANGE_FAILED",
                    "error": str(exc),
                    "committed_patch_count": 0,
                    "round_count": 0,
                    "committed_patches": [],
                }
            post, post_states = _support_reading(
                problem,
                post_indices,
                target_root_count=int(target_root_count),
                compiled_hamiltonian=compiled_hamiltonian,
            )
            post["matching_to_pre_exchange"] = _state_matching(pre_states, post_states)
            selector_round = (
                dict(selector_rounds[growth_round - 1])
                if growth_round - 1 < len(selector_rounds)
                else None
            )
            if selector_round is not None:
                screening = dict(selector_round.get("screening") or {})
                screening_totals["selection_round_count"] += 1
                for field in screening_count_fields:
                    screening_totals[field] += int(screening.get(field, 0))
                acquisition_work, acquisition_word_sets = _selection_acquisition_work(
                    problem,
                    order[: growth_round - 1],
                    screening,
                    hamiltonian_words=hamiltonian_words,
                )
                for stage, words in acquisition_word_sets.items():
                    acquisition_words_by_stage[stage].update(words)
                event_sources = {
                    "retained_pencil": acquisition_work["retained_pencil"],
                    "residualized_2x2_screen": acquisition_work[
                        "residualized_2x2_screen"
                    ],
                    "ky_fan_rerank_raw": acquisition_work["ky_fan_rerank"][
                        "raw_candidate_rows"
                    ],
                }
                for stage, source in event_sources.items():
                    acquisition_event_totals[stage]["pair_events"] += int(
                        source.get("pair_count", 0)
                    )
                    acquisition_event_totals[stage]["naive_term_events"] += int(
                        source.get("naive_terms", 0)
                    )
                    acquisition_event_totals[stage][
                        "candidate_compiled_K_event_sum"
                    ] += float(source.get("candidate_compiled_K_sum", 0.0))
                acquisition_cumulative = _cumulative_acquisition_payload(
                    problem,
                    acquisition_words_by_stage,
                    acquisition_event_totals,
                )
            else:
                acquisition_work = None
                acquisition_cumulative = _cumulative_acquisition_payload(
                    problem,
                    acquisition_words_by_stage,
                    acquisition_event_totals,
                )
            trace.append(
                {
                    "growth_round": int(growth_round),
                    "selected_record": {
                        "index": int(selected_index),
                        "name": str(problem.basis[int(selected_index)].name),
                        "score": decisions[int(selected_index)].get("selection_score"),
                        "geometry": dict(decisions[int(selected_index)].get("geometry") or {}),
                    },
                    "selector_round": selector_round,
                    "selection_screening_cumulative": dict(screening_totals),
                    "selection_acquisition_work": acquisition_work,
                    "selection_acquisition_cumulative": acquisition_cumulative,
                    "pre_exchange": pre,
                    "exchange": exchange,
                    "post_exchange": post,
                }
            )
        profile_payload = {
            "score_mode": profile.score_mode,
            "additive_novelty_reward": profile.additive_novelty_reward,
            "novelty_floor": float(profile.selector_config.geometry_min_metric_novelty),
            "selection_config": static_record_selection_payload(selection)[
                "selection_config"
            ],
            "selection_order": order,
            "selector_stop": dict(selection.geometry_stop or {}),
            "selection_screening_totals": dict(screening_totals),
            "selection_acquisition_totals": _cumulative_acquisition_payload(
                problem,
                acquisition_words_by_stage,
                acquisition_event_totals,
            ),
            "growth_rounds": [int(row["growth_round"]) for row in trace],
            "trace": trace,
        }
        if profile.profile_id in shortlist_profile_ids:
            shortlist_payload[profile.profile_id] = profile_payload
        else:
            profiles_payload[profile.profile_id] = profile_payload
    references = [float(value) for value in problem.references[: int(target_root_count)]]
    exact_boundary_gap = (
        float(problem.references[int(target_root_count)] - problem.references[int(target_root_count) - 1])
        if len(problem.references) > int(target_root_count)
        else None
    )
    return {
        "schema_version": "paper_iii_qse_score_factorial_v2",
        "policy": "diagnostic_only_selector_identity_factorial",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "problem": {
            "regime": str(problem.regime),
            "n_ph_max": int(problem.n_ph_max),
            "pool_digest": str(problem.pool_digest),
            "reference_key": str(problem.reference_key),
            "pool_size": len(problem.basis),
        },
        "target_root_count": int(target_root_count),
        "reference_root_energies": references,
        "reference_ky_fan_sum": float(sum(references)),
        "reference_gap_R_to_R_plus_1": exact_boundary_gap,
        "requested_growth_cap": int(cap),
        "growth_rounds": sorted(
            {
                int(row["growth_round"])
                for profile_payload in (
                    *profiles_payload.values(),
                    *shortlist_payload.values(),
                )
                for row in profile_payload["trace"]
            }
        ),
        "profile_count": len(profiles_payload),
        "profiles": profiles_payload,
        "shortlist_profile_count": len(shortlist_payload),
        "shortlist_profiles": shortlist_payload,
    }


def run_factorial_score_ablation_campaign(
    problems: Sequence[PaperIIIProblem],
    *,
    k_max: int,
    target_root_count: int = _TARGET_ROOTS,
    exchange_max_rounds: int = 4,
    shortlist_sizes: Sequence[int] = (),
    regime_set: str = "custom",
    method_lock: bool = False,
    profile_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the locked factorial for an already materialized problem sequence."""

    requested_shortlist_sizes = tuple(int(value) for value in shortlist_sizes)
    if method_lock:
        if profile_ids is not None:
            raise ValueError("method_lock requires its complete declared profile set.")
        principled_sizes = (int(target_root_count), 2 * int(target_root_count))
        if requested_shortlist_sizes and requested_shortlist_sizes != principled_sizes:
            raise ValueError(
                "method_lock fixes shortlist sizes to (R, 2R); do not supply "
                "different shortlist_sizes."
            )
        resolved_shortlist_sizes = principled_sizes
    else:
        resolved_shortlist_sizes = requested_shortlist_sizes
    if profile_ids is None:
        executed_shortlist_sizes = resolved_shortlist_sizes
    else:
        requested_profile_id_set = {str(value) for value in profile_ids}
        executed_shortlist_sizes = tuple(
            size
            for size in resolved_shortlist_sizes
            if (
                f"{GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST}"
                f"__shortlist_{size}"
            )
            in requested_profile_id_set
        )
    cases = [
        run_factorial_score_ablation(
            problem,
            k_max=int(k_max),
            target_root_count=int(target_root_count),
            exchange_max_rounds=int(exchange_max_rounds),
            shortlist_sizes=resolved_shortlist_sizes,
            profile_ids=profile_ids,
        )
        for problem in problems
    ]
    payload = {
        "schema_version": "paper_iii_qse_score_factorial_campaign_v1",
        "policy": "diagnostic_only_selector_identity_factorial",
        "regime_set": str(regime_set),
        "settings": {
            "k_max": int(k_max),
            "target_root_count": int(target_root_count),
            "exchange_max_rounds": int(exchange_max_rounds),
            "shortlist_sizes": [int(value) for value in executed_shortlist_sizes],
        },
        "case_count": len(cases),
        "cases": cases,
    }
    if method_lock:
        payload["settings"]["method_lock"] = True
        payload["method_lock"] = _method_lock_assessment(
            cases,
            target_root_count=int(target_root_count),
        )
    elif profile_ids is not None:
        payload["settings"]["profile_ids"] = [str(value) for value in profile_ids]
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regime-set", choices=sorted(REGIME_SETS), default="nph1")
    parser.add_argument(
        "--regimes",
        default=None,
        help="Comma-separated regime filter within the selected cheap-tier set.",
    )
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--target-root-count", type=int, default=_TARGET_ROOTS)
    parser.add_argument("--exchange-max-rounds", type=int, default=4)
    parser.add_argument(
        "--shortlist-sizes",
        default="2,4,8",
        help="Comma-separated residualized-2x2 shortlist sizes; empty disables them.",
    )
    parser.add_argument(
        "--method-lock",
        action="store_true",
        help=(
            "replace ad hoc shortlist sizes by m=R and m=2R and emit the "
            "pre-exchange one-growth-round lock decision"
        ),
    )
    parser.add_argument(
        "--profile-ids",
        default=None,
        help="Comma-separated exact profile IDs for a narrow validation run.",
    )
    args = parser.parse_args(argv)

    wanted = (
        None
        if args.regimes is None
        else {token.strip() for token in str(args.regimes).split(",") if token.strip()}
    )
    specs = [
        spec
        for spec in REGIME_SETS[str(args.regime_set)]
        if wanted is None or str(spec[0]) in wanted
    ]
    if not specs:
        parser.error("the regime filter selected no cases")
    try:
        shortlist_sizes = tuple(
            int(token.strip())
            for token in str(args.shortlist_sizes).split(",")
            if token.strip()
        )
    except ValueError:
        parser.error("--shortlist-sizes must be a comma-separated list of integers")
    if any(value < 1 for value in shortlist_sizes):
        parser.error("--shortlist-sizes entries must be positive")
    if args.method_lock:
        shortlist_sizes = (
            int(args.target_root_count),
            2 * int(args.target_root_count),
        )
    profile_ids = (
        None
        if args.profile_ids is None
        else tuple(
            token.strip()
            for token in str(args.profile_ids).split(",")
            if token.strip()
        )
    )
    if profile_ids == ():
        parser.error("--profile-ids selected no profiles")
    if args.method_lock and profile_ids is not None:
        parser.error("--method-lock cannot be combined with --profile-ids")
    problems: list[PaperIIIProblem] = []
    for regime, u, g_ep, n_ph_max in specs:
        problem = load_problem(
            regime=str(regime),
            u=float(u),
            g_ep=float(g_ep),
            n_ph_max=int(n_ph_max),
            target_roots=int(args.target_root_count),
        )
        problems.append(problem)
        profile_summary = (
            f"profiles={list(profile_ids)}"
            if profile_ids is not None
            else f"factorial=8 shortlist={list(shortlist_sizes)}"
        )
        print(
            f"== {regime} (nph{n_ph_max}, pool={len(problem.basis)}) "
            f"{profile_summary} k_max={args.k_max}",
            flush=True,
        )

    payload = run_factorial_score_ablation_campaign(
        problems,
        k_max=int(args.k_max),
        target_root_count=int(args.target_root_count),
        exchange_max_rounds=int(args.exchange_max_rounds),
        shortlist_sizes=shortlist_sizes,
        regime_set=str(args.regime_set),
        method_lock=bool(args.method_lock),
        profile_ids=profile_ids,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output_json: {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
