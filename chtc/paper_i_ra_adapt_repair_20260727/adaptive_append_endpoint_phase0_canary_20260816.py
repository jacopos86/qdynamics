"""Inert append-endpoint graph-weighted Phase-0 diagnostic implementation.

This module is deliberately separate from the canonical RA-ADAPT controller.
It specifies and tests reuse of one append-endpoint gradient per available
generator, a graph-proxy denominator, one fixed-24 shadow arm, and one active
adaptive-shortlist arm.  It is not execution authority: installation refuses
until the corrected native Phase-I--III semantic-closure route and digests own
the complete run.  Later insertion-position policy is outside this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any, Mapping, Sequence

from pipelines.static_adapt.ra_adapt.adaptive_append_endpoint_shortlist import (
    AppendEndpointGeneratorScore,
    select_adaptive_append_endpoint_shortlist,
)


MODE_FIXED24_SHADOW = "fixed24_graph_weighted_adaptive_shadow_v1"
MODE_ACTIVE_ADAPTIVE = "active_adaptive_graph_weighted_v1"
ADAPTIVE_APPEND_ENDPOINT_PHASE0_MODES = frozenset(
    {MODE_FIXED24_SHADOW, MODE_ACTIVE_ADAPTIVE}
)
TEMPORARY_OVERLAY_EXECUTION_AUTHORIZED = False
ADAPTIVE_APPEND_ENDPOINT_PHASE0_POLICY = (
    "append_endpoint_graph_weighted_phase0_diagnostic_v1"
)
ADAPTIVE_APPEND_ENDPOINT_PHASE0_RECEIPT_SCHEMA = (
    "paper_i_append_endpoint_graph_weighted_phase0_receipt_v1"
)
ADAPTIVE_APPEND_ENDPOINT_PHASE0_CONSUMER_SCOPE = (
    "phase0_append_endpoint_generator_gradient_surface_v1"
)
ADAPTIVE_APPEND_ENDPOINT_PHASE0_POPULATION_SCOPE = (
    "current_available_append_endpoint_generators_v1"
)
PARENT_ABSOLUTE_GRADIENT_PHASE0_POLICY = (
    "global_singleton_absolute_gradient_shortlist_v1"
)
PARENT_GLOBAL_SINGLETON_PHASE0_RECEIPT_SCHEMA = (
    "paper_i_global_singleton_gradient_phase0_receipt_v1"
)
_COST_COMPONENTS = ("2q", "d", "1q", "theta", "shot")


def canonical_sha256(payload: Any) -> str:
    """Return the repository's canonical JSON digest."""

    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _same_float(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    except (TypeError, ValueError):
        return False


def _normalize_scored_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        pool_index = int(row.get("pool_index", -1))
        append_position = int(row.get("append_position", -1))
        generator_id = str(row.get("generator_id", ""))
        pool_label = str(row.get("pool_label", ""))
        gradient = _finite(
            row.get("append_gradient_signed"),
            label="Append-endpoint gradient",
        )
        denominator = _finite(
            row.get("graph_proxy_denominator"),
            label="Graph-proxy denominator",
        )
        source = str(row.get("graph_proxy_source", ""))
        if (
            pool_index < 0
            or append_position < 0
            or not generator_id
            or not pool_label
            or denominator <= 0.0
            or source != "proxy_logical_ladder_span_v1"
        ):
            raise ValueError("Append-endpoint scored-row identity or source is invalid.")
        raw_cost = dict(row.get("graph_proxy_raw", {}))
        bars = dict(row.get("graph_proxy_bars", {}))
        if set(raw_cost) != set(_COST_COMPONENTS) or set(bars) != set(
            _COST_COMPONENTS
        ):
            raise ValueError("Graph-proxy cost components are incomplete.")
        raw_cost = {
            key: _finite(raw_cost[key], label=f"Raw graph cost {key}")
            for key in _COST_COMPONENTS
        }
        bars = {
            key: _finite(bars[key], label=f"Normalized graph cost {key}")
            for key in _COST_COMPONENTS
        }
        if any(value < 0.0 for value in (*raw_cost.values(), *bars.values())):
            raise ValueError("Graph-proxy cost components must be nonnegative.")
        excess = _finite(
            row.get("graph_proxy_cost_excess_sum"),
            label="Graph-proxy excess",
        )
        if excess < 0.0:
            raise ValueError("Graph-proxy excess must be nonnegative.")
        normalized.append(
            {
                "pool_index": pool_index,
                "generator_id": generator_id,
                "pool_label": pool_label,
                "append_position": append_position,
                "append_gradient_signed": gradient,
                "graph_proxy_source": source,
                "graph_proxy_raw": raw_cost,
                "graph_proxy_bars": bars,
                "graph_proxy_cost_excess_sum": excess,
                "graph_proxy_denominator": denominator,
            }
        )
    normalized.sort(key=lambda row: int(row["pool_index"]))
    indices = [int(row["pool_index"]) for row in normalized]
    generator_ids = [str(row["generator_id"]) for row in normalized]
    positions = {int(row["append_position"]) for row in normalized}
    if (
        not normalized
        or len(set(indices)) != len(indices)
        or len(set(generator_ids)) != len(generator_ids)
        or len(positions) != 1
    ):
        raise ValueError("Append-endpoint scored population identity is invalid.")
    return normalized


def _validate_graph_proxy_normalization(
    rows: Sequence[Mapping[str, Any]],
    normalization: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(normalization)
    normalization_rows = payload.get("rows")
    denominators = payload.get("denominators")
    medians = dict(payload.get("medians", {}))
    scales = dict(payload.get("scales", {}))
    lambdas = dict(payload.get("lambdas", {}))
    scale_floor = _finite(
        payload.get("scale_floor"),
        label="Graph-proxy normalization scale floor",
    )
    if (
        payload.get("schema")
        != "snake_hardware_cost_candidate_record_denominator_v1"
        or payload.get("scope") != "candidate_records"
        or payload.get("normalization_schema")
        != "snake_hardware_cost_family_robust_v1"
        or not str(payload.get("lambda_source", ""))
        or scale_floor <= 0.0
        or set(medians) != set(_COST_COMPONENTS)
        or set(scales) != set(_COST_COMPONENTS)
        or set(lambdas) != set(_COST_COMPONENTS)
        or not isinstance(normalization_rows, list)
        or not isinstance(denominators, list)
        or len(normalization_rows) != len(rows)
        or len(denominators) != len(rows)
    ):
        raise ValueError("Graph-proxy normalization contract is invalid.")
    median_values = {
        key: _finite(medians[key], label=f"Graph-proxy median {key}")
        for key in _COST_COMPONENTS
    }
    scale_values = {
        key: _finite(scales[key], label=f"Graph-proxy scale {key}")
        for key in _COST_COMPONENTS
    }
    lambda_values = {
        key: _finite(lambdas[key], label=f"Graph-proxy lambda {key}")
        for key in _COST_COMPONENTS
    }
    if any(value < 0.0 for value in median_values.values()) or any(
        value < scale_floor for value in scale_values.values()
    ):
        raise ValueError("Graph-proxy normalization scale is invalid.")
    for key in _COST_COMPONENTS:
        raw_values = [float(row["graph_proxy_raw"][key]) for row in rows]
        expected_median = float(statistics.median(raw_values))
        positive_excesses = [
            value - expected_median
            for value in raw_values
            if value > expected_median
        ]
        expected_scale = max(
            scale_floor,
            float(statistics.median(positive_excesses))
            if positive_excesses
            else scale_floor,
        )
        if not _same_float(median_values[key], expected_median) or not _same_float(
            scale_values[key], expected_scale
        ):
            raise ValueError("Graph-proxy normalization statistics drifted.")
    for index, (row, raw_norm_row, raw_denominator) in enumerate(
        zip(rows, normalization_rows, denominators)
    ):
        if not isinstance(raw_norm_row, Mapping):
            raise ValueError("Graph-proxy normalization row is malformed.")
        norm_row = dict(raw_norm_row)
        expected_bars = {
            key: float(
                math.asinh(
                    max(
                        0.0,
                        float(row["graph_proxy_raw"][key])
                        - median_values[key],
                    )
                    / scale_values[key]
                )
            )
            for key in _COST_COMPONENTS
        }
        expected_excess = float(
            max(
                0.0,
                sum(
                    lambda_values[key] * expected_bars[key]
                    for key in _COST_COMPONENTS
                ),
            )
        )
        expected_denominator = float(max(1.0, 1.0 + expected_excess))
        if (
            int(norm_row.get("index", -1)) != index
            or str(norm_row.get("label", "")) != str(row["pool_label"])
            or int(norm_row.get("candidate_pool_index", -1))
            != int(row["pool_index"])
            or int(norm_row.get("position_id", -1))
            != int(row["append_position"])
            or dict(norm_row.get("raw", {})) != dict(row["graph_proxy_raw"])
            or any(
                not _same_float(
                    dict(norm_row.get("bars", {})).get(key),
                    expected_bars[key],
                )
                or not _same_float(row["graph_proxy_bars"][key], expected_bars[key])
                for key in _COST_COMPONENTS
            )
            or not _same_float(
                norm_row.get("hardware_cost_excess_sum"), expected_excess
            )
            or not _same_float(
                row["graph_proxy_cost_excess_sum"], expected_excess
            )
            or not _same_float(
                norm_row.get("hardware_cost_denominator"), expected_denominator
            )
            or not _same_float(raw_denominator, expected_denominator)
            or not _same_float(
                row["graph_proxy_denominator"], expected_denominator
            )
        ):
            raise ValueError("Graph-proxy normalization row drifted.")
    return payload


def filter_position_domain_by_retained_generators(
    admissible_domain: Sequence[Any],
    *,
    ranked_pool_indices: Sequence[int],
    retained_pool_indices: Sequence[int],
) -> tuple[Any, ...]:
    """Keep every downstream position belonging to a retained generator."""

    population = tuple(admissible_domain)
    ranked = tuple(int(value) for value in ranked_pool_indices)
    retained = tuple(int(value) for value in retained_pool_indices)
    if not population or len(set(ranked)) != len(ranked):
        raise ValueError("Append-endpoint Phase-0 domain identity is invalid.")
    if len(set(retained)) != len(retained) or not set(retained).issubset(ranked):
        raise ValueError("Append-endpoint Phase-0 retained identity is invalid.")
    domain_indices = {int(record.pool_index) for record in population}
    if domain_indices != set(ranked):
        raise ValueError(
            "Append-endpoint Phase-0 generator population differs from its "
            "immutable position domain."
        )
    retained_set = set(retained)
    rank_by_pool = {pool_index: rank for rank, pool_index in enumerate(ranked)}
    return tuple(
        sorted(
            (
                record
                for record in population
                if int(record.pool_index) in retained_set
            ),
            key=lambda record: (
                rank_by_pool[int(record.pool_index)],
                int(record.insertion_position),
                str(record.domain_record_id),
            ),
        )
    )


def select_append_endpoint_phase0_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    cap: int = 24,
) -> dict[str, Any]:
    """Select one active generator shortlist and retain the adaptive decision."""

    selected_mode = str(mode)
    if selected_mode not in ADAPTIVE_APPEND_ENDPOINT_PHASE0_MODES:
        raise ValueError("Unknown append-endpoint Phase-0 diagnostic mode.")
    if isinstance(cap, bool):
        raise ValueError("Append-endpoint Phase-0 cap must be an integer, not bool.")
    cap_value = int(cap)
    if cap_value < 1:
        raise ValueError("Append-endpoint Phase-0 cap must be positive.")
    normalized = [dict(row) for row in rows]
    if not normalized:
        raise ValueError("Append-endpoint Phase-0 population must be non-empty.")
    scores: list[AppendEndpointGeneratorScore] = []
    for row in normalized:
        pool_index = int(row.get("pool_index", -1))
        gradient = float(row.get("append_gradient_signed"))
        denominator = float(row.get("graph_proxy_denominator"))
        if pool_index < 0:
            raise ValueError("Append-endpoint Phase-0 pool index is invalid.")
        if not math.isfinite(gradient):
            raise ValueError("Append-endpoint Phase-0 gradient must be finite.")
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                "Append-endpoint graph-proxy denominator must be positive and finite."
            )
        scores.append(
            AppendEndpointGeneratorScore(
                generator_index=pool_index,
                append_gradient=gradient,
                graph_cost=denominator,
            )
        )
    adaptive = select_adaptive_append_endpoint_shortlist(scores, cap=cap_value)
    adaptive_receipt = adaptive.to_receipt()
    ranked = list(adaptive.ranked_generator_indices)
    if selected_mode == MODE_FIXED24_SHADOW:
        positive_ranked = [
            int(row["generator_index"])
            for row in adaptive_receipt["ranking"]
            if float(row["utility"]) > 0.0
        ]
        retained = positive_ranked[: min(cap_value, len(positive_ranked))]
        active_policy = "fixed_top_k_by_utility_v1"
        adaptive_role = "shadow"
    else:
        retained = list(adaptive.retained_generator_indices)
        active_policy = "adaptive_effective_competition_v1"
        adaptive_role = "active"
    return {
        "mode": selected_mode,
        "cap": cap_value,
        "active_shortlist_policy": active_policy,
        "adaptive_decision_role": adaptive_role,
        "status": "stationary" if not retained else "competitive",
        "ranked_pool_indices": ranked,
        "retained_pool_indices": retained,
        "adaptive_decision": adaptive_receipt,
    }


def build_append_endpoint_phase0_receipt(
    rows: Sequence[Mapping[str, Any]],
    *,
    graph_proxy_normalization: Mapping[str, Any],
    estimator_event_ids: Sequence[str],
    mode: str,
    cap: int = 24,
    parent_policy: str = PARENT_ABSOLUTE_GRADIENT_PHASE0_POLICY,
    parent_receipt_schema: str = PARENT_GLOBAL_SINGLETON_PHASE0_RECEIPT_SCHEMA,
    parent_consumer_scope: str = "global_singleton_gradient_phase0",
    parent_population_scope: str = "current_available_global_guarded_singletons_v1",
) -> dict[str, Any]:
    """Build one self-digesting, independently recomputable Phase-0 receipt."""

    normalized = _normalize_scored_rows(rows)
    normalization = _validate_graph_proxy_normalization(
        normalized,
        graph_proxy_normalization,
    )
    decision = select_append_endpoint_phase0_rows(
        normalized,
        mode=str(mode),
        cap=cap,
    )
    event_ids = [str(value) for value in estimator_event_ids]
    if (
        len(event_ids) != len(normalized)
        or len(set(event_ids)) != len(event_ids)
        or any(not value for value in event_ids)
    ):
        raise ValueError(
            "Append-endpoint Phase-0 estimator events do not close N_grad."
        )
    adaptive_ranking = {
        int(row["generator_index"]): dict(row)
        for row in decision["adaptive_decision"]["ranking"]
    }
    retained_set = set(int(value) for value in decision["retained_pool_indices"])
    adaptive_retained = set(
        int(value)
        for value in decision["adaptive_decision"]["retained_generator_indices"]
    )
    row_by_pool = {int(row["pool_index"]): dict(row) for row in normalized}
    ranking: list[dict[str, Any]] = []
    for rank, pool_index in enumerate(decision["ranked_pool_indices"], start=1):
        population_row = row_by_pool[int(pool_index)]
        adaptive_row = adaptive_ranking[int(pool_index)]
        ranking.append(
            {
                **population_row,
                "append_gradient_abs": float(
                    abs(float(population_row["append_gradient_signed"]))
                ),
                "utility": float(adaptive_row["utility"]),
                "rank": int(rank),
                "active_retained": int(pool_index) in retained_set,
                "adaptive_retained": int(pool_index) in adaptive_retained,
            }
        )
    components = {
        "N_H_outer": 0,
        "N_H_refit": 0,
        "N_grad": len(normalized),
        "N_metric": 0,
    }
    zero_components = {
        "N_H_outer": 0,
        "N_H_refit": 0,
        "N_grad": 0,
        "N_metric": 0,
    }
    payload: dict[str, Any] = {
        "schema": ADAPTIVE_APPEND_ENDPOINT_PHASE0_RECEIPT_SCHEMA,
        "policy": ADAPTIVE_APPEND_ENDPOINT_PHASE0_POLICY,
        "mode": str(decision["mode"]),
        "parent_policy": str(parent_policy),
        "parent_receipt_schema": str(parent_receipt_schema),
        "parent_consumer_scope": str(parent_consumer_scope),
        "parent_population_scope": str(parent_population_scope),
        "population_scope": ADAPTIVE_APPEND_ENDPOINT_PHASE0_POPULATION_SCOPE,
        "consumer_scope": ADAPTIVE_APPEND_ENDPOINT_PHASE0_CONSUMER_SCOPE,
        "gradient_surface": "append_endpoint_generators_v1",
        "position_aware_gradient_surface": False,
        "insertion_position_scope": (
            "append_endpoint_generator_screen_before_downstream_position_policy_v1"
        ),
        "downstream_insertion_policy": "independent_unmodified_v1",
        "score": "squared_append_gradient_over_graph_proxy_denominator_v1",
        "ranking_order": "descending_utility_then_pool_index_v1",
        "graph_proxy_cost_policy": "family_robust_positive_denominator_v1",
        "graph_proxy_compile_source": "phase1_logical_graph_proxy_v1",
        "qiskit_compile_cost_policy": "off",
        "qiskit_compile_cost_scope": "phase0_only_v1",
        "metric_policy": "off",
        "measurement_cost_policy": "off",
        "semantic_ownership_scope": (
            "phase0_append_endpoint_shortlist_only_v1"
        ),
        "later_phase_semantics_ownership": (
            "external_source_locked_route_contract_v1"
        ),
        "later_phase_qiskit_semantics_claimed": False,
        "later_phase_zero_centered_semantics_claimed": False,
        "execution_authorized": False,
        "native_semantic_closure_required": True,
        "execution_authority": "none_inert_implementation_only_v1",
        "requested_cap": int(decision["cap"]),
        "active_shortlist_policy": str(decision["active_shortlist_policy"]),
        "adaptive_decision_role": str(decision["adaptive_decision_role"]),
        "status": str(decision["status"]),
        "append_position": int(normalized[0]["append_position"]),
        "input_candidate_count": len(normalized),
        "retained_candidate_count": len(retained_set),
        "effective_shortlist_size": len(retained_set),
        "input_pool_indices": [int(row["pool_index"]) for row in normalized],
        "ranked_pool_indices": [
            int(value) for value in decision["ranked_pool_indices"]
        ],
        "retained_pool_indices": [
            int(value) for value in decision["retained_pool_indices"]
        ],
        "input_population_sha256": canonical_sha256(normalized),
        "retained_population_sha256": canonical_sha256(
            [
                row
                for row in normalized
                if int(row["pool_index"]) in retained_set
            ]
        ),
        "population": normalized,
        "ranking": ranking,
        "graph_proxy_normalization": normalization,
        "adaptive_decision": dict(decision["adaptive_decision"]),
        "estimator_event_ids": event_ids,
        "estimator_accounting": {
            "unit": "executed_logical_scalar_estimator_invocation",
            "components": components,
            **components,
            "S_alg": int(sum(components.values())),
            "zero_metric_measurements": True,
        },
        "adaptive_shadow_accounting": {
            "source": "classical_reuse_of_active_gradient_and_proxy_population_v1",
            "components": zero_components,
            **zero_components,
            "S_alg": 0,
        },
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _validate_scored_position_projection(
    receipt: Mapping[str, Any],
    scored_population: Mapping[str, Any],
) -> None:
    screen = scored_population.get("phase0_gradient_screen")
    if not isinstance(screen, Mapping) or screen.get("schema") != (
        "paper_i_scored_gradient_phase0_population_v1"
    ):
        raise RuntimeError("Append-endpoint Phase-0 scored domain is absent.")
    population = screen.get("population")
    shortlist = screen.get("shortlist")
    if (
        not isinstance(population, list)
        or not population
        or any(not isinstance(row, Mapping) for row in population)
        or not isinstance(shortlist, list)
        or any(not isinstance(row, Mapping) for row in shortlist)
        or int(screen.get("population_count", -1)) != len(population)
        or int(screen.get("shortlist_count", -1)) != len(shortlist)
        or screen.get("ordered_population_sha256") != canonical_sha256(population)
        or screen.get("ordered_shortlist_sha256") != canonical_sha256(shortlist)
    ):
        raise RuntimeError("Append-endpoint Phase-0 scored domain is malformed.")
    input_indices = set(int(value) for value in receipt["input_pool_indices"])
    ranked_indices = [int(value) for value in receipt["ranked_pool_indices"]]
    retained_indices = set(int(value) for value in receipt["retained_pool_indices"])
    if {int(row.get("pool_index", -1)) for row in population} != input_indices:
        raise RuntimeError("Append-endpoint Phase-0 population domain drifted.")
    rank_by_pool = {
        pool_index: rank for rank, pool_index in enumerate(ranked_indices)
    }
    expected_shortlist = sorted(
        (
            dict(row)
            for row in population
            if int(row.get("pool_index", -1)) in retained_indices
        ),
        key=lambda row: (
            rank_by_pool[int(row["pool_index"])],
            int(row["insertion_position"]),
            str(row["domain_record_id"]),
        ),
    )
    if [dict(row) for row in shortlist] != expected_shortlist:
        raise RuntimeError(
            "Append-endpoint Phase-0 changed the retained generator's position domain."
        )


def validate_append_endpoint_phase0_receipt(
    raw_receipt: Mapping[str, Any],
    *,
    scored_population: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deep-validate one receipt by rebuilding every selection decision."""

    try:
        receipt = dict(raw_receipt)
        observed_sha256 = receipt.get("sha256")
        unsigned = dict(receipt)
        unsigned.pop("sha256", None)
        if observed_sha256 != canonical_sha256(unsigned):
            raise RuntimeError("Append-endpoint Phase-0 receipt digest is invalid.")
        expected = build_append_endpoint_phase0_receipt(
            receipt.get("population", []),
            graph_proxy_normalization=receipt.get(
                "graph_proxy_normalization", {}
            ),
            estimator_event_ids=receipt.get("estimator_event_ids", []),
            mode=str(receipt.get("mode", "")),
            cap=receipt.get("requested_cap", 0),
            parent_policy=str(receipt.get("parent_policy", "")),
            parent_receipt_schema=str(
                receipt.get("parent_receipt_schema", "")
            ),
            parent_consumer_scope=str(
                receipt.get("parent_consumer_scope", "")
            ),
            parent_population_scope=str(
                receipt.get("parent_population_scope", "")
            ),
        )
        if receipt != expected:
            raise RuntimeError(
                "Append-endpoint Phase-0 receipt failed decision recomputation."
            )
        if scored_population is not None:
            _validate_scored_position_projection(receipt, scored_population)
    except RuntimeError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "Append-endpoint Phase-0 receipt failed decision recomputation."
        ) from exc
    return dict(raw_receipt)


def _available_pool_indices(cursor: Any) -> tuple[int, ...]:
    raw = getattr(cursor, "available_indices", None)
    if raw is None:
        provider = getattr(cursor, "selection_available_indices", None)
        raw = provider() if callable(provider) else None
    if raw is None:
        raise RuntimeError("Append-endpoint Phase-0 available population is absent.")
    indices = tuple(sorted(int(value) for value in raw))
    if not indices or len(set(indices)) != len(indices):
        raise RuntimeError("Append-endpoint Phase-0 available population is invalid.")
    return indices


def execute_adaptive_append_endpoint_phase0(
    transaction: Any,
    *,
    pipeline_module: Any,
    admissible_domain: Sequence[Any],
    shortlist_size: int,
    policy: str,
    receipt_schema: str,
    consumer_scope: str,
    population_scope: str,
    mode: str,
) -> Any:
    """Execute the named overlay through one source-locked transaction seam."""

    from pipelines.scaffold.hh_continuation_scoring import (
        hardware_cost_candidate_record_denominators,
    )

    if (
        isinstance(shortlist_size, bool)
        or int(shortlist_size) != 24
        or str(policy) != PARENT_ABSOLUTE_GRADIENT_PHASE0_POLICY
        or str(receipt_schema) != PARENT_GLOBAL_SINGLETON_PHASE0_RECEIPT_SCHEMA
        or str(consumer_scope) != "phase0_global_singleton_gradient_surface"
        or str(population_scope)
        != "current_available_global_guarded_singletons_v1"
    ):
        raise RuntimeError("Append-endpoint Phase-0 parent route binding drifted.")
    pending = transaction.pending
    context = transaction.context
    cursor = transaction.cursor
    if pending.phase0_gradient_shortlist_receipt is not None:
        raise RuntimeError("Append-endpoint Phase 0 executed more than once.")
    if (
        getattr(context.transition_services, "controller_noise_runtime", None)
        is not None
    ):
        raise RuntimeError("Append-endpoint adaptive Phase 0 is exact/noiseless only.")

    available = _available_pool_indices(cursor)
    if set(available) != set(int(value) for value in pending.available_sorted):
        raise RuntimeError("Append-endpoint Phase-0 available population drifted.")
    occurrence_start = len(transaction.ledger_occurrences())
    transaction.session._evaluate_default_candidate_gradient_surface(
        pending,
        consumer_scope=str(consumer_scope),
    )
    transaction.session._refresh_default_candidate_gradient_summaries(pending)
    occurrences = transaction.ledger_occurrences()[occurrence_start:]
    if len(occurrences) != len(available) or any(
        row.get("component") != "N_grad"
        or row.get("consumer_scope") != str(consumer_scope)
        or not isinstance(row.get("sequence"), int)
        or not str(row.get("primitive_id", ""))
        for row in occurrences
    ):
        raise RuntimeError(
            "Append-endpoint Phase 0 did not close one gradient per generator."
        )
    event_ids = [
        f"estimator:{int(row['sequence'])}:{str(row['primitive_id'])}"
        for row in occurrences
    ]
    if len(set(event_ids)) != len(event_ids):
        raise RuntimeError("Append-endpoint Phase-0 gradient events repeat.")

    population = tuple(admissible_domain)
    identity_by_pool: dict[int, tuple[str, str]] = {}
    for record in population:
        pool_index = int(record.pool_index)
        identity = (str(record.generator_id), str(record.pool_label))
        previous = identity_by_pool.setdefault(pool_index, identity)
        if previous != identity:
            raise RuntimeError(
                "Append-endpoint Phase-0 generator identity changed by position."
            )
    if set(identity_by_pool) != set(available):
        raise RuntimeError(
            "Append-endpoint Phase-0 population differs from its position domain."
        )

    append_position = int(pending.append_position)
    nested_window = pipeline_module._predict_nested_refit_window_for_position(
        theta=pending.theta_logical_current,
        position_id=append_position,
        policy=context.reoptimization_policy,
        window_size=context.reoptimization_window_size,
        window_topk=context.reoptimization_window_topk,
        periodic_full_refit_triggered=False,
    )
    nested_accounting = pipeline_module.build_nested_window_accounting(
        nested_window,
        compile_proxy_basis=str(
            getattr(
                pipeline_module,
                "COMPILE_PROXY_BASIS_OLD_PRE_INHERITED",
                "old_pre_inherited",
            )
        ),
    )
    compile_rows: list[dict[str, Any]] = []
    compile_sources: list[str] = []
    for pool_index in available:
        candidate = context.pool[pool_index]
        compiled_candidate = context.compiled_pool[pool_index]
        estimate = context.phase1_compile_oracle.estimate(
            candidate_term_count=int(len(compiled_candidate.terms)),
            position_id=append_position,
            append_position=append_position,
            refit_active_count=int(nested_accounting.compile_proxy_refit_count),
            candidate_term=candidate,
        )
        source = str(getattr(estimate, "hardware_cost_source", ""))
        source_mode = str(getattr(estimate, "source_mode", ""))
        if source != "proxy_logical_ladder_span_v1" or source_mode != "proxy":
            raise RuntimeError(
                "Append-endpoint Phase 0 attempted a non-graph compile source."
            )
        values = {
            "c_hat_2q": _finite(
                getattr(estimate, "c_hat_2q", None), label="Graph proxy 2q"
            ),
            "c_hat_d": _finite(
                getattr(estimate, "c_hat_d", None), label="Graph proxy depth"
            ),
            "c_hat_1q": _finite(
                getattr(estimate, "c_hat_1q", None), label="Graph proxy 1q"
            ),
            "c_hat_theta": _finite(
                getattr(estimate, "c_hat_theta", None),
                label="Graph proxy theta",
            ),
            "c_hat_shot": 0.0,
        }
        if any(value < 0.0 for value in values.values()):
            raise RuntimeError("Append-endpoint graph-proxy cost is negative.")
        compile_rows.append(
            {
                "label": str(identity_by_pool[pool_index][1]),
                "candidate_pool_index": int(pool_index),
                "position_id": append_position,
                **values,
            }
        )
        compile_sources.append(source)
    normalization = hardware_cost_candidate_record_denominators(
        compile_rows,
        pending.phase2_score_cfg_round,
    )
    normalized_rows = normalization.get("rows")
    denominators = normalization.get("denominators")
    if (
        not isinstance(normalized_rows, list)
        or not isinstance(denominators, list)
        or len(normalized_rows) != len(available)
        or len(denominators) != len(available)
    ):
        raise RuntimeError("Append-endpoint graph-proxy normalization is incomplete.")
    scored_rows: list[dict[str, Any]] = []
    for offset, pool_index in enumerate(available):
        norm_row = dict(normalized_rows[offset])
        scored_rows.append(
            {
                "pool_index": int(pool_index),
                "generator_id": str(identity_by_pool[pool_index][0]),
                "pool_label": str(identity_by_pool[pool_index][1]),
                "append_position": append_position,
                "append_gradient_signed": float(pending.gradients[pool_index]),
                "graph_proxy_source": str(compile_sources[offset]),
                "graph_proxy_raw": dict(norm_row.get("raw", {})),
                "graph_proxy_bars": dict(norm_row.get("bars", {})),
                "graph_proxy_cost_excess_sum": float(
                    norm_row.get("hardware_cost_excess_sum")
                ),
                "graph_proxy_denominator": float(denominators[offset]),
            }
        )
    receipt = build_append_endpoint_phase0_receipt(
        scored_rows,
        graph_proxy_normalization=normalization,
        estimator_event_ids=event_ids,
        mode=str(mode),
        cap=int(shortlist_size),
        parent_policy=str(policy),
        parent_receipt_schema=str(receipt_schema),
        parent_consumer_scope=str(consumer_scope),
        parent_population_scope=str(population_scope),
    )
    pending.phase0_gradient_shortlist_receipt = receipt
    retained = [int(value) for value in receipt["retained_pool_indices"]]
    pending.shortlist[:] = retained
    if not retained:
        raise RuntimeError(
            "Append-endpoint graph-weighted Phase 0 is stationary; downstream "
            "Phase I was not entered."
        )
    shortlisted_records = filter_position_domain_by_retained_generators(
        population,
        ranked_pool_indices=receipt["ranked_pool_indices"],
        retained_pool_indices=retained,
    )
    utility_by_pool = {
        int(row["pool_index"]): float(row["utility"])
        for row in receipt["ranking"]
    }
    ranking = tuple(
        pipeline_module._ShortlistRankReceipt(
            record_key=(str(record.domain_record_id), str(record.generator_id)),
            shortlist_rank=int(rank),
            primary_score=float(utility_by_pool[int(record.pool_index)]),
            tie_break_score=float(utility_by_pool[int(record.pool_index)]),
            pool_index=int(record.pool_index),
            insertion_position=int(record.insertion_position),
        )
        for rank, record in enumerate(shortlisted_records, start=1)
    )
    return pipeline_module._PhaseSelectionReceipt(
        phase="phase0",
        population=population,
        shortlist=shortlisted_records,
        shortlist_ranking=ranking,
        estimator_event_ids=tuple(event_ids),
    )


def install_adaptive_append_endpoint_phase0_overlay(
    *,
    mode: str,
    pipeline_module: Any | None = None,
    engine_module: Any | None = None,
) -> Any:
    """Refuse temporary installation pending the native semantic-closure route."""

    selected_mode = str(mode)
    if selected_mode not in ADAPTIVE_APPEND_ENDPOINT_PHASE0_MODES:
        raise ValueError("Unknown append-endpoint Phase-0 diagnostic mode.")
    del pipeline_module, engine_module
    raise RuntimeError(
        "Temporary append-endpoint overlay is inert; install the versioned "
        "native semantic-closure route before any scientific execution."
    )


__all__ = [
    "ADAPTIVE_APPEND_ENDPOINT_PHASE0_CONSUMER_SCOPE",
    "ADAPTIVE_APPEND_ENDPOINT_PHASE0_MODES",
    "ADAPTIVE_APPEND_ENDPOINT_PHASE0_POLICY",
    "ADAPTIVE_APPEND_ENDPOINT_PHASE0_POPULATION_SCOPE",
    "ADAPTIVE_APPEND_ENDPOINT_PHASE0_RECEIPT_SCHEMA",
    "MODE_ACTIVE_ADAPTIVE",
    "MODE_FIXED24_SHADOW",
    "PARENT_ABSOLUTE_GRADIENT_PHASE0_POLICY",
    "PARENT_GLOBAL_SINGLETON_PHASE0_RECEIPT_SCHEMA",
    "TEMPORARY_OVERLAY_EXECUTION_AUTHORIZED",
    "build_append_endpoint_phase0_receipt",
    "canonical_sha256",
    "execute_adaptive_append_endpoint_phase0",
    "filter_position_domain_by_retained_generators",
    "install_adaptive_append_endpoint_phase0_overlay",
    "select_append_endpoint_phase0_rows",
    "validate_append_endpoint_phase0_receipt",
]
