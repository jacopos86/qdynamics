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
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.paper_iii_matched_accuracy_campaign import REGIME_SETS
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
) -> dict[str, Any]:
    """Run the eight-cell factorial plus optional shortlist profiles."""

    cap = min(int(k_max), len(problem.basis))
    if cap < 1:
        raise ValueError("k_max must admit at least one growth round.")
    if int(target_root_count) < 1:
        raise ValueError("target_root_count must be positive.")
    compiled_hamiltonian = compile_polynomial_action(problem.hamiltonian)
    profiles_payload: dict[str, Any] = {}
    shortlist_payload: dict[str, Any] = {}
    factorial_profiles = factorial_score_profiles(
        max_records=len(problem.basis), target_root_count=int(target_root_count)
    )
    shortlist_profiles = shortlist_score_profiles(
        shortlist_sizes=shortlist_sizes,
        max_records=len(problem.basis),
        target_root_count=int(target_root_count),
    )
    for profile in (*factorial_profiles, *shortlist_profiles):
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
            "growth_rounds": [int(row["growth_round"]) for row in trace],
            "trace": trace,
        }
        if profile in shortlist_profiles:
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
) -> dict[str, Any]:
    """Run the locked factorial for an already materialized problem sequence."""

    cases = [
        run_factorial_score_ablation(
            problem,
            k_max=int(k_max),
            target_root_count=int(target_root_count),
            exchange_max_rounds=int(exchange_max_rounds),
            shortlist_sizes=tuple(int(value) for value in shortlist_sizes),
        )
        for problem in problems
    ]
    return {
        "schema_version": "paper_iii_qse_score_factorial_campaign_v1",
        "policy": "diagnostic_only_selector_identity_factorial",
        "regime_set": str(regime_set),
        "settings": {
            "k_max": int(k_max),
            "target_root_count": int(target_root_count),
            "exchange_max_rounds": int(exchange_max_rounds),
            "shortlist_sizes": [int(value) for value in shortlist_sizes],
        },
        "case_count": len(cases),
        "cases": cases,
    }


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
        print(
            f"== {regime} (nph{n_ph_max}, pool={len(problem.basis)}) "
            f"factorial=8 shortlist={list(shortlist_sizes)} k_max={args.k_max}",
            flush=True,
        )

    payload = run_factorial_score_ablation_campaign(
        problems,
        k_max=int(args.k_max),
        target_root_count=int(args.target_root_count),
        exchange_max_rounds=int(args.exchange_max_rounds),
        shortlist_sizes=shortlist_sizes,
        regime_set=str(args.regime_set),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output_json: {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
