"""Diagnostic position-aware Phase-0 overlay for the Paper-I Page-12 route.

This module does not change the canonical RA-ADAPT implementation.  It patches
one extracted, source-locked worker process for the explicitly authorized
strong--weak always-open k=15 canary.  Phase 0 evaluates the standard
coordinate energy gradient for every commutation-reduced
``(generator, insertion-position)`` record and retains the best 24 records.
The later Phase-I/II/III implementation is unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


POSITION_PHASE0_POLICY = (
    "global_singleton_insertion_position_absolute_gradient_shortlist_v1"
)
POSITION_PHASE0_SCHEMA = (
    "paper_i_global_singleton_insertion_position_gradient_phase0_receipt_v1"
)
POSITION_PHASE0_CONSUMER_SCOPE = (
    "phase0_global_singleton_insertion_position_gradient_surface"
)
POSITION_PHASE0_POPULATION_SCOPE = (
    "current_commutation_reduced_global_guarded_singleton_position_records_v1"
)
POSITION_PHASE0_SCORE = "absolute_coordinate_energy_gradient_at_position_v1"
POSITION_PHASE0_INSERTION_SCOPE = (
    "all_commutation_reduced_insertion_position_records_before_shortlist_v1"
)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _record_projection(record: Any, *, append_position: int) -> dict[str, Any]:
    position = int(record.insertion_position)
    return {
        "domain_record_id": str(record.domain_record_id),
        "generator_id": str(record.generator_id),
        "pool_index": int(record.pool_index),
        "pool_label": str(record.pool_label),
        "insertion_position": position,
        "position_class": "interior" if position < int(append_position) else "append",
    }


def rank_position_gradient_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    shortlist_size: int,
) -> list[dict[str, Any]]:
    """Return the deterministic top-k position records by ``|gradient|``."""

    cap = int(shortlist_size)
    if cap < 1:
        raise ValueError("Position-aware Phase-0 shortlist must be positive.")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for raw in rows:
        row = dict(raw)
        gradient = float(row.get("gradient_signed"))
        key = (
            int(row.get("pool_index", -1)),
            int(row.get("insertion_position", -1)),
            str(row.get("domain_record_id", "")),
        )
        if min(key[:2]) < 0 or not key[2] or key in seen:
            raise ValueError("Position-aware Phase-0 row identity is invalid.")
        if not math.isfinite(gradient):
            raise ValueError("Position-aware Phase-0 gradients must be finite.")
        seen.add(key)
        normalized.append(
            {
                **row,
                "gradient_signed": gradient,
                "gradient_abs": float(abs(gradient)),
            }
        )
    if not normalized:
        raise ValueError("Position-aware Phase-0 population must be non-empty.")
    ranked = sorted(
        normalized,
        key=lambda row: (
            -float(row["gradient_abs"]),
            int(row["pool_index"]),
            int(row["insertion_position"]),
            str(row["domain_record_id"]),
        ),
    )
    return ranked[: min(cap, len(ranked))]


def filtered_position_plans(
    plans: Mapping[int, Mapping[str, Any]],
    retained_positions_by_pool: Mapping[int, Sequence[int]],
) -> dict[int, dict[str, Any]]:
    """Project full commutation plans onto retained representative records."""

    projected: dict[int, dict[str, Any]] = {}
    for raw_pool_index, raw_positions in retained_positions_by_pool.items():
        pool_index = int(raw_pool_index)
        source = plans.get(pool_index)
        if not isinstance(source, Mapping):
            raise ValueError("Retained Phase-0 pool index has no position plan.")
        representatives = sorted({int(value) for value in raw_positions})
        source_representatives = {
            int(value) for value in source.get("representative_positions", [])
        }
        if not representatives or not set(representatives).issubset(
            source_representatives
        ):
            raise ValueError("Retained Phase-0 position escaped its source plan.")
        raw_members = source.get("members_by_representative", {})
        members_by_representative: dict[int, list[int]] = {}
        representative_by_position: dict[int, int] = {}
        requested: list[int] = []
        for representative in representatives:
            members = sorted(
                {int(value) for value in raw_members.get(representative, [])}
            )
            if not members or representative != min(members):
                raise ValueError("Retained commutation class is malformed.")
            members_by_representative[representative] = members
            requested.extend(members)
            for position in members:
                representative_by_position[position] = representative
        requested = sorted(set(requested))
        projected[pool_index] = {
            "schema": str(source.get("schema")),
            "requested_positions": requested,
            "representative_positions": representatives,
            "representative_by_position": representative_by_position,
            "members_by_representative": members_by_representative,
            "commuting_crossings": [
                bool(value) for value in source.get("commuting_crossings", [])
            ],
            "collapsed_position_count": int(len(requested) - len(representatives)),
        }
    if not projected:
        raise ValueError("Position-aware Phase 0 retained no position plans.")
    return projected


def _validate_position_phase0_receipt(
    engine: Any,
    row: Mapping[str, Any],
    *,
    scored_population: Mapping[str, Any],
) -> dict[str, Any]:
    raw = row.get("ra_gradient_phase0_shortlist")
    if not isinstance(raw, Mapping):
        raise RuntimeError("Accepted round is missing position-aware Phase-0 evidence.")
    receipt = dict(raw)
    observed_digest = receipt.pop("sha256", None)
    accounting = receipt.get("estimator_accounting")
    components = accounting.get("components") if isinstance(accounting, Mapping) else None
    event_ids = receipt.get("estimator_event_ids")
    ranking = receipt.get("ranking")
    input_count = int(receipt.get("input_candidate_count", -1))
    retained_count = int(receipt.get("retained_candidate_count", -1))
    screen = scored_population.get("phase0_gradient_screen")
    if (
        receipt.get("schema") != POSITION_PHASE0_SCHEMA
        or receipt.get("policy") != POSITION_PHASE0_POLICY
        or observed_digest != engine.canonical_sha256(receipt)
        or receipt.get("score") != POSITION_PHASE0_SCORE
        or receipt.get("metric_policy") != "off"
        or receipt.get("compile_cost_policy") != "off"
        or receipt.get("measurement_cost_policy") != "off"
        or receipt.get("insertion_position_scope") != POSITION_PHASE0_INSERTION_SCOPE
        or int(receipt.get("requested_shortlist_size", -1)) != 24
        or retained_count != min(24, input_count)
        or not isinstance(event_ids, list)
        or len(event_ids) != input_count
        or len(set(str(value) for value in event_ids)) != input_count
        or not isinstance(ranking, list)
        or len(ranking) != retained_count
        or components
        != {
            "N_H_outer": 0,
            "N_H_refit": 0,
            "N_grad": input_count,
            "N_metric": 0,
        }
        or accounting.get("S_alg") != input_count
        or accounting.get("zero_metric_measurements") is not True
        or not isinstance(screen, Mapping)
        or screen.get("schema") != "paper_i_scored_gradient_phase0_population_v1"
        or int(screen.get("population_count", -1)) != input_count
        or int(screen.get("shortlist_count", -1)) != retained_count
        or receipt.get("input_population_sha256")
        != screen.get("ordered_population_sha256")
        or receipt.get("retained_population_sha256")
        != screen.get("ordered_shortlist_sha256")
    ):
        raise RuntimeError("Accepted position-aware gradient Phase-0 evidence is invalid.")
    shortlist = screen.get("shortlist")
    if not isinstance(shortlist, list) or any(
        not isinstance(value, Mapping) for value in shortlist
    ):
        raise RuntimeError("Position-aware Phase-0 shortlist is malformed.")
    projected_ranking = [
        {
            key: rank_row.get(key)
            for key in (
                "domain_record_id",
                "generator_id",
                "pool_index",
                "pool_label",
                "insertion_position",
                "position_class",
            )
        }
        for rank_row in ranking
    ]
    if projected_ranking != [dict(value) for value in shortlist]:
        raise RuntimeError("Position-aware Phase-0 ranking and shortlist disagree.")
    magnitudes = [float(value.get("gradient_abs")) for value in ranking]
    if any(not math.isfinite(value) or value < 0.0 for value in magnitudes) or any(
        left < right for left, right in zip(magnitudes, magnitudes[1:])
    ):
        raise RuntimeError("Position-aware Phase-0 ranking order is invalid.")
    return dict(raw)


def install_position_aware_phase0_overlay() -> Callable[[], None]:
    """Patch the active extracted worker modules and return a restore callback."""

    from pipelines.static_adapt import adapt_pipeline as pipeline
    from pipelines.static_adapt.ra_adapt import engine

    transaction_type = pipeline._DefaultNoPruneSelectionTransaction
    original_phase0 = transaction_type.run_absolute_gradient_phase0
    original_phase1 = transaction_type.run_phase_i
    original_validator = engine._validated_gradient_phase0_round_receipt
    retained_state: dict[int, dict[str, Any]] = {}

    def position_phase0(
        self: Any,
        *,
        admissible_domain: tuple[Any, ...],
        shortlist_size: int,
        policy: str,
        receipt_schema: str,
        consumer_scope: str,
        population_scope: str,
    ) -> Any:
        del receipt_schema, consumer_scope, population_scope
        pending = self.pending
        if pending.phase0_gradient_shortlist_receipt is not None:
            raise RuntimeError("Position-aware gradient Phase 0 executed twice.")
        if pending.insertion_mode != "full_commutation_reduced":
            raise RuntimeError("Position-aware Phase 0 is authorized only for always-open insertion.")
        if int(shortlist_size) != 24:
            raise RuntimeError("Position-aware Phase 0 requires the source cap of 24.")
        if str(policy) != "global_singleton_absolute_gradient_shortlist_v1":
            raise RuntimeError("Position-aware Phase 0 lost its parent policy binding.")
        if (
            getattr(
                self.session.context.transition_services,
                "controller_noise_runtime",
                None,
            )
            is not None
        ):
            raise RuntimeError("The position-aware canary is exact/noiseless only.")

        theta = np.asarray(pending.theta_logical_current, dtype=float)
        geometry_context = pipeline._prepare_exact_insertion_first_order_context(
            selected_ops=list(self.cursor.selected_ops),
            theta=theta,
            psi_ref=np.asarray(self.context.reference_state, dtype=complex),
            psi_state=np.asarray(pending.psi_current, dtype=complex),
            hpsi_state=np.asarray(pending.hpsi_current, dtype=complex),
            pauli_action_cache=self.cursor.pauli_action_cache,
            state_consistency_tolerance=float(
                max(1.0e-12, pending.phase2_score_cfg_round.batch_state_consistency_tolerance)
            ),
        )
        occurrence_start = len(self.ledger_occurrences())
        population_rows: list[dict[str, Any]] = []
        gradient_by_record_id: dict[str, float] = {}
        for record in admissible_domain:
            pool_index = int(record.pool_index)
            position = int(record.insertion_position)
            candidate = self.context.pool[pool_index]
            geometry = pipeline._exact_insertion_first_order_candidate_geometry(
                context=geometry_context,
                candidate_term=candidate,
                position_id=position,
                candidate_compiled=self.context.compiled_pool[pool_index],
                pauli_action_cache=self.cursor.pauli_action_cache,
            )
            gradient = float(geometry["energy_gradient"])
            if not math.isfinite(gradient):
                raise RuntimeError("Position-aware Phase-0 gradient is non-finite.")
            service = self.session.estimator_service
            service._record_estimator_primitive(
                state=np.asarray(pending.psi_current, dtype=complex),
                component="N_grad",
                consumer_scope=POSITION_PHASE0_CONSUMER_SCOPE,
                primitive_kind="coordinate_gradient",
                observable_or_formula_identity="coordinate_energy_gradient_v2",
                operand_identity=service._candidate_physical_tangent(
                    list(self.cursor.selected_ops),
                    theta,
                    candidate,
                    insertion_position=position,
                ),
            )
            projection = _record_projection(
                record,
                append_position=int(pending.append_position),
            )
            population_rows.append({**projection, "gradient_signed": gradient})
            gradient_by_record_id[str(record.domain_record_id)] = gradient

        occurrences = self.ledger_occurrences()[occurrence_start:]
        if len(occurrences) != len(admissible_domain) or any(
            row.get("component") != "N_grad"
            or row.get("consumer_scope") != POSITION_PHASE0_CONSUMER_SCOPE
            for row in occurrences
        ):
            raise RuntimeError("Position-aware Phase 0 did not close its gradient ledger.")
        retained_rows = rank_position_gradient_rows(
            population_rows,
            shortlist_size=int(shortlist_size),
        )
        retained_keys = {
            (int(row["pool_index"]), int(row["insertion_position"]))
            for row in retained_rows
        }
        shortlisted_records = tuple(
            record
            for record in admissible_domain
            if (int(record.pool_index), int(record.insertion_position)) in retained_keys
        )
        rank_by_key = {
            (int(row["pool_index"]), int(row["insertion_position"])): rank
            for rank, row in enumerate(retained_rows, start=1)
        }
        shortlisted_records = tuple(
            sorted(
                shortlisted_records,
                key=lambda record: rank_by_key[
                    (int(record.pool_index), int(record.insertion_position))
                ],
            )
        )
        if len(shortlisted_records) != len(retained_rows):
            raise RuntimeError("Position-aware Phase-0 shortlist identity drifted.")

        retained_positions_by_pool: dict[int, list[int]] = {}
        ordered_pool_indices: list[int] = []
        for row in retained_rows:
            pool_index = int(row["pool_index"])
            if pool_index not in retained_positions_by_pool:
                retained_positions_by_pool[pool_index] = []
                ordered_pool_indices.append(pool_index)
            retained_positions_by_pool[pool_index].append(
                int(row["insertion_position"])
            )
        original_plans = pending.candidate_position_plans
        retained_state[id(pending)] = {
            "full_plans": original_plans,
            "filtered_plans": filtered_position_plans(
                original_plans,
                retained_positions_by_pool,
            ),
        }
        pending.shortlist[:] = ordered_pool_indices

        population_projection = [
            {key: row[key] for key in (
                "domain_record_id", "generator_id", "pool_index", "pool_label",
                "insertion_position", "position_class",
            )}
            for row in population_rows
        ]
        retained_projection = [
            {key: row[key] for key in (
                "domain_record_id", "generator_id", "pool_index", "pool_label",
                "insertion_position", "position_class",
            )}
            for row in retained_rows
        ]
        event_ids = [
            f"estimator:{int(row['sequence'])}:{str(row['primitive_id'])}"
            for row in occurrences
        ]
        components = {
            "N_H_outer": 0,
            "N_H_refit": 0,
            "N_grad": len(occurrences),
            "N_metric": 0,
        }
        receipt: dict[str, Any] = {
            "schema": POSITION_PHASE0_SCHEMA,
            "policy": POSITION_PHASE0_POLICY,
            "parent_policy": str(policy),
            "population_scope": POSITION_PHASE0_POPULATION_SCOPE,
            "score": POSITION_PHASE0_SCORE,
            "ranking_order": (
                "descending_absolute_gradient_then_pool_index_then_position_v1"
            ),
            "metric_policy": "off",
            "compile_cost_policy": "off",
            "measurement_cost_policy": "off",
            "lane_policy": "single_global_population_v1",
            "insertion_position_scope": POSITION_PHASE0_INSERTION_SCOPE,
            "requested_shortlist_size": int(shortlist_size),
            "effective_shortlist_size": len(retained_rows),
            "input_candidate_count": len(population_rows),
            "retained_candidate_count": len(retained_rows),
            "input_population_sha256": canonical_sha256(population_projection),
            "retained_population_sha256": canonical_sha256(retained_projection),
            "ranking": [
                {**row, "rank": rank}
                for rank, row in enumerate(retained_rows, start=1)
            ],
            "estimator_event_ids": event_ids,
            "estimator_accounting": {
                "unit": "executed_logical_scalar_estimator_invocation",
                "components": components,
                **components,
                "S_alg": int(sum(components.values())),
                "zero_metric_measurements": True,
            },
        }
        receipt["sha256"] = canonical_sha256(receipt)
        pending.phase0_gradient_shortlist_receipt = receipt

        ranking_receipts = tuple(
            pipeline._ShortlistRankReceipt(
                record_key=(str(record.domain_record_id), str(record.generator_id)),
                shortlist_rank=int(rank),
                primary_score=float(abs(gradient_by_record_id[str(record.domain_record_id)])),
                tie_break_score=float(abs(gradient_by_record_id[str(record.domain_record_id)])),
                pool_index=int(record.pool_index),
                insertion_position=int(record.insertion_position),
            )
            for rank, record in enumerate(shortlisted_records, start=1)
        )
        return pipeline._PhaseSelectionReceipt(
            phase="phase0",
            population=tuple(admissible_domain),
            shortlist=shortlisted_records,
            shortlist_ranking=ranking_receipts,
            estimator_event_ids=tuple(event_ids),
        )

    def filtered_phase1(self: Any) -> dict[str, Any]:
        state = retained_state.get(id(self.pending))
        if state is None:
            return original_phase1(self)
        full_plans = state["full_plans"]
        self.pending.candidate_position_plans = state["filtered_plans"]
        try:
            result = original_phase1(self)
        finally:
            self.pending.candidate_position_plans = full_plans
        score_eval = result.get("score_eval")
        if not isinstance(score_eval, Mapping):
            raise RuntimeError("Position-aware Phase I omitted its score surface.")
        full_domain_receipt = pipeline._always_commutation_reduced_domain_receipt(
            candidate_position_plans=full_plans,
            pool=self.context.pool,
        )
        score_eval["insertion_commutation_reduced"] = full_domain_receipt
        result["positions_considered"] = list(
            range(int(self.pending.append_position) + 1)
        )
        result["insertion_probe_triggered"] = bool(
            int(self.pending.append_position) > 0
        )
        result["insertion_probe_reason"] = (
            "position_aware_phase0_full_commutation_reduced"
        )
        return result

    def position_validator(
        row: Mapping[str, Any],
        *,
        scored_population: Mapping[str, Any],
        algorithm_id: str,
    ) -> dict[str, Any]:
        raw = row.get("ra_gradient_phase0_shortlist")
        if isinstance(raw, Mapping) and raw.get("schema") == POSITION_PHASE0_SCHEMA:
            return _validate_position_phase0_receipt(
                engine,
                row,
                scored_population=scored_population,
            )
        return original_validator(
            row,
            scored_population=scored_population,
            algorithm_id=algorithm_id,
        )

    transaction_type.run_absolute_gradient_phase0 = position_phase0
    transaction_type.run_phase_i = filtered_phase1
    engine._validated_gradient_phase0_round_receipt = position_validator

    def restore() -> None:
        transaction_type.run_absolute_gradient_phase0 = original_phase0
        transaction_type.run_phase_i = original_phase1
        engine._validated_gradient_phase0_round_receipt = original_validator
        retained_state.clear()

    return restore


__all__ = [
    "POSITION_PHASE0_CONSUMER_SCOPE",
    "POSITION_PHASE0_INSERTION_SCOPE",
    "POSITION_PHASE0_POLICY",
    "POSITION_PHASE0_SCHEMA",
    "filtered_position_plans",
    "install_position_aware_phase0_overlay",
    "rank_position_gradient_rows",
]
