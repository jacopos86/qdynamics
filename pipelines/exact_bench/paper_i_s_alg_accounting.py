"""Paper-I clean-algorithm logical scalar-estimator accounting.

``S_alg`` counts the logical scalar estimator invocations required by the
declared algorithm through a displayed accepted prefix.  It is not a count of
unique physical identities: repeated optimizer objectives count again.
Conversely, implementation-only duplicate bridges, superseded parent
Phase-III geometry, post-prefix diagnostics, and explicitly carried
same-iteration Phase-I/II/III data do not count.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


PAPER_I_S_ALG_ACCOUNTING_SCHEMA = "paper_i_clean_algorithm_s_alg_v3"
PAPER_I_S_ALG_CONTRACT = (
    "required_executed_logical_scalar_estimator_invocations_v1"
)
PROJECTED_PHASE_ORDER = (
    "phase1_parent_shortlist_then_split_then_phase2_children_then_phase3"
)
S_ALG_COMPONENTS = ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
SNAKE_REPRESENTATION_INTACT_MACRO = "intact_macro"
SNAKE_REPRESENTATION_PROJECTED_SINGLETON = "projected_singleton"
_SNAKE_REPRESENTATIONS = frozenset(
    {
        SNAKE_REPRESENTATION_INTACT_MACRO,
        SNAKE_REPRESENTATION_PROJECTED_SINGLETON,
    }
)


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, not Boolean.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a nonnegative integer.") from exc
    if parsed < 0 or (
        isinstance(value, float) and float(value) != float(parsed)
    ):
        raise ValueError(f"{field} must be a nonnegative integer.")
    return int(parsed)


def _component_mapping(
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, int]:
    components = payload.get("components")
    if not isinstance(components, Mapping):
        raise ValueError(f"{field}.components is missing.")
    normalized = {
        component: _nonnegative_int(
            components.get(component),
            field=f"{field}.components.{component}",
        )
        for component in S_ALG_COMPONENTS
    }
    return normalized


def _work_receipt(
    *,
    method: str,
    representation: str,
    accepted_prefix_length: int,
    components: Mapping[str, int],
    normalization: Mapping[str, Any],
    round_cardinalities: Sequence[Mapping[str, int]] = (),
) -> dict[str, Any]:
    normalized = {
        component: _nonnegative_int(
            components.get(component),
            field=f"components.{component}",
        )
        for component in S_ALG_COMPONENTS
    }
    s_alg = int(sum(normalized.values()))
    return {
        "schema": PAPER_I_S_ALG_ACCOUNTING_SCHEMA,
        "contract": PAPER_I_S_ALG_CONTRACT,
        "method": str(method),
        "representation": str(representation),
        "accepted_prefix_length": int(accepted_prefix_length),
        "scope": (
            "all required estimator invocations through the displayed "
            "post-admission prefix; post-prefix diagnostics excluded"
        ),
        "unit": "logical_scalar_estimator_invocation",
        "components": normalized,
        "S_alg": int(s_alg),
        "normalization": dict(normalization),
        "round_cardinalities": [dict(row) for row in round_cardinalities],
    }


def _controller_event_scope(
    controller: Mapping[str, Any],
    *,
    history_index: int,
    phase: str,
    event_kind: str,
) -> Mapping[str, Any]:
    by_scope = controller.get("by_scope")
    if not isinstance(by_scope, Mapping):
        raise ValueError(
            f"history[{history_index}] lacks controller by_scope receipts."
        )
    phase_token = f"|phase={phase}|"
    event_token = f"|event={event_kind}|"
    matches = [
        value
        for key, value in by_scope.items()
        if phase_token in str(key) and event_token in str(key)
    ]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise ValueError(
            f"history[{history_index}] expected exactly one {phase}/"
            f"{event_kind} controller scope, found {len(matches)}."
        )
    return matches[0]


def snake_round_cardinality_from_history(
    row: Mapping[str, Any],
    *,
    history_index: int,
    representation: str,
) -> dict[str, int]:
    """Extract one independently cross-checked SNAKE phase cardinality row.

    The Paper-I projected-singleton route shortlists macro parents in Phase I,
    expands those retained parents into singleton children, and sends the
    child population into Phase II.  The signed ``scored_surface_records``
    inventory closes the complete generated child population, including
    children rejected by the later child shortlist.
    """

    index = _nonnegative_int(history_index, field="history_index")
    representation_key = str(representation).strip().lower()
    if representation_key not in _SNAKE_REPRESENTATIONS:
        raise ValueError(
            f"representation must be one of {sorted(_SNAKE_REPRESENTATIONS)!r}."
        )
    n_active_raw = row.get("phase3_active_logical_coordinate_count")
    if n_active_raw is None:
        checkpoint = row.get("active_prefix_checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(
                f"history[{index}] lacks both the Phase-III active-coordinate "
                "count and an accepted-prefix checkpoint."
            )
        accepted_depth = _nonnegative_int(
            checkpoint.get("active_ansatz_depth"),
            field=(
                f"history[{index}].active_prefix_checkpoint."
                "active_ansatz_depth"
            ),
        )
        n_active_raw = accepted_depth - 1
    n_active = _nonnegative_int(
        n_active_raw,
        field=(
            f"history[{index}].phase3_active_logical_coordinate_count"
        ),
    )
    if n_active != index:
        raise ValueError(
            f"history[{index}] is not a one-admission, no-prune prefix: "
            f"n_active={n_active}, expected={index}."
        )
    controller = row.get("controller_measurement_work_proxy")
    if not isinstance(controller, Mapping):
        raise ValueError(
            f"history[{index}] lacks controller_measurement_work_proxy."
        )
    by_phase = controller.get("by_phase")
    if not isinstance(by_phase, Mapping):
        raise ValueError(
            f"history[{index}] lacks by_phase controller receipts."
        )
    phase1 = by_phase.get("phase1")
    phase2 = by_phase.get("phase2")
    if not isinstance(phase1, Mapping) or not isinstance(phase2, Mapping):
        raise ValueError(
            f"history[{index}] lacks Phase-I/II controller receipts."
        )
    r1 = _nonnegative_int(
        phase1.get("method_input_candidate_count_total"),
        field=(
            f"history[{index}].controller.by_phase.phase1."
            "method_input_candidate_count_total"
        ),
    )
    phase1_operator_probe_count = _nonnegative_int(
        phase1.get("actual_operator_probe_count_total"),
        field=(
            f"history[{index}].controller.by_phase.phase1."
            "actual_operator_probe_count_total"
        ),
    )
    if phase1_operator_probe_count != r1:
        raise ValueError(
            f"history[{index}] Phase-I logical-estimator input does not close "
            f"to the operator-probe receipt: R1={r1}, "
            f"actual_operator_probe_count={phase1_operator_probe_count}."
        )
    r2 = _nonnegative_int(
        phase1.get("method_shortlist_candidate_count_total"),
        field=(
            f"history[{index}].controller.by_phase.phase1."
            "method_shortlist_candidate_count_total"
        ),
    )
    phase2_events = _nonnegative_int(
        phase2.get("events_count", 0),
        field=f"history[{index}].controller.by_phase.phase2.events_count",
    )
    phase2_scope = _controller_event_scope(
        controller,
        history_index=index,
        phase="phase2",
        event_kind="phase2_rerank_records",
    )
    for field_name in (
        "method_input_candidate_count_total",
        "actual_evaluated_candidate_count_total",
        "candidate_count_total",
        "pre_shortlist_count_total",
        "records_evaluated",
    ):
        observed = _nonnegative_int(
            phase2_scope.get(field_name),
            field=(
                f"history[{index}].controller.by_scope.phase2_rerank_records."
                f"{field_name}"
            ),
        )
        if (
            representation_key == SNAKE_REPRESENTATION_INTACT_MACRO
            and observed != r2
        ):
            raise ValueError(
                f"history[{index}] Phase-II input does not close to the "
                f"Phase-I shortlist: R2={r2}, {field_name}={observed}."
            )

    output: dict[str, int] = {
        "history_index": int(index),
        "n_active": int(n_active),
        "R1": int(r1),
        "R2": int(r2),
        "phase2_acquisition_event_count": int(phase2_events),
    }
    phase3 = by_phase.get("phase3")
    if representation_key == SNAKE_REPRESENTATION_INTACT_MACRO:
        if not isinstance(phase3, Mapping):
            raise ValueError(
                f"history[{index}] lacks the Phase-III controller receipt."
            )
        r3 = _nonnegative_int(
            phase3.get("method_input_candidate_count_total"),
            field=(
                f"history[{index}].controller.by_phase.phase3."
                "method_input_candidate_count_total"
            ),
        )
        phase3_scope = _controller_event_scope(
            controller,
            history_index=index,
            phase="phase3",
            event_kind="phase3_reduced_geometry_rerank",
        )
        phase3_audit = {}
        for field_name in (
            "method_input_candidate_count_total",
            "candidate_count_total",
            "actual_evaluated_candidate_count_total",
            "pre_shortlist_count_total",
            "records_evaluated",
        ):
            observed = _nonnegative_int(
                phase3_scope.get(field_name),
                field=(
                    f"history[{index}].controller.by_scope."
                    "phase3_reduced_geometry_rerank."
                    f"{field_name}"
                ),
            )
            phase3_audit[field_name] = observed
        if any(observed != r3 for observed in phase3_audit.values()):
            raise ValueError(
                f"history[{index}] intact-macro Phase-III evaluated input "
                f"does not close: R3={r3}, audit={phase3_audit}."
            )
        output.update(
            {
                "R3": int(r3),
                "R3_evaluated_candidate_count": int(r3),
            }
        )
        return output

    projected_receipt = row.get("projected_phase3_population_receipt")
    if (
        isinstance(projected_receipt, Mapping)
        and projected_receipt.get("schema")
        == "paper_i_projected_phase3_population_receipt_v2"
    ):
        expected_phase_order = (
            "phase1_parent_shortlist_then_split_then_"
            "phase2_children_then_phase3"
        )
        if projected_receipt.get("phase_order") != expected_phase_order:
            raise ValueError(
                f"history[{index}] projected receipt has the wrong phase order."
            )
        retained_parent_count = _nonnegative_int(
            projected_receipt.get("phase1_retained_parent_count"),
            field=(
                f"history[{index}].projected_receipt."
                "phase1_retained_parent_count"
            ),
        )
        child_population = _nonnegative_int(
            projected_receipt.get("phase2_input_child_count"),
            field=(
                f"history[{index}].projected_receipt."
                "phase2_input_child_count"
            ),
        )
        phase2_retained_child_count = _nonnegative_int(
            projected_receipt.get("phase2_retained_child_count"),
            field=(
                f"history[{index}].projected_receipt."
                "phase2_retained_child_count"
            ),
        )
        phase3_child_count = _nonnegative_int(
            projected_receipt.get("phase3_evaluated_candidate_count"),
            field=(
                f"history[{index}].projected_receipt."
                "phase3_evaluated_candidate_count"
            ),
        )
        if retained_parent_count != r2:
            raise ValueError(
                f"history[{index}] projected Phase-I parent receipt does not "
                f"close: controller={r2}, receipt={retained_parent_count}."
            )
        if (
            child_population < 1
            or phase2_retained_child_count < 1
            or phase2_retained_child_count > child_population
            or phase3_child_count != phase2_retained_child_count
        ):
            raise ValueError(
                f"history[{index}] projected child populations do not close."
            )
        for field_name in (
            "method_input_candidate_count_total",
            "actual_evaluated_candidate_count_total",
            "candidate_count_total",
            "pre_shortlist_count_total",
            "records_evaluated",
        ):
            observed = _nonnegative_int(
                phase2_scope.get(field_name),
                field=(
                    f"history[{index}].controller.by_scope."
                    f"phase2_rerank_records.{field_name}"
                ),
            )
            if observed != child_population:
                raise ValueError(
                    f"history[{index}] projected Phase-II child input does "
                    f"not close: children={child_population}, "
                    f"{field_name}={observed}."
                )
        if not isinstance(phase3, Mapping):
            raise ValueError(
                f"history[{index}] lacks the projected Phase-III receipt."
            )
        controller_phase3_count = _nonnegative_int(
            phase3.get("method_input_candidate_count_total"),
            field=(
                f"history[{index}].controller.by_phase.phase3."
                "method_input_candidate_count_total"
            ),
        )
        phase3_scope = _controller_event_scope(
            controller,
            history_index=index,
            phase="phase3",
            event_kind="phase3_reduced_geometry_rerank",
        )
        for field_name in (
            "method_input_candidate_count_total",
            "candidate_count_total",
            "actual_evaluated_candidate_count_total",
            "pre_shortlist_count_total",
            "records_evaluated",
        ):
            observed = _nonnegative_int(
                phase3_scope.get(field_name),
                field=(
                    f"history[{index}].controller.by_scope."
                    f"phase3_reduced_geometry_rerank.{field_name}"
                ),
            )
            if observed != phase3_child_count:
                raise ValueError(
                    f"history[{index}] projected Phase-III child input does "
                    f"not close: children={phase3_child_count}, "
                    f"{field_name}={observed}."
                )
        if controller_phase3_count != phase3_child_count:
            raise ValueError(
                f"history[{index}] projected Phase-III controller count "
                "does not close."
            )
        split_parent_count = _nonnegative_int(
            projected_receipt.get("split_parent_count", 0),
            field=(
                f"history[{index}].projected_receipt.split_parent_count"
            ),
        )
        split_child_count = _nonnegative_int(
            projected_receipt.get("split_child_count", 0),
            field=(
                f"history[{index}].projected_receipt.split_child_count"
            ),
        )
        unsplit_singleton_count = _nonnegative_int(
            projected_receipt.get("unsplit_singleton_count", 0),
            field=(
                f"history[{index}].projected_receipt."
                "unsplit_singleton_count"
            ),
        )
        output.update(
            {
                "phase1_retained_parent_count": int(
                    retained_parent_count
                ),
                "R2": int(child_population),
                "R3": int(phase3_child_count),
                "R2_split_parent_count": int(split_parent_count),
                "R2_split_child_count": int(split_child_count),
                "R2_unsplit_singleton_count": int(
                    unsplit_singleton_count
                ),
                "R2_evaluated_child_count": int(child_population),
                "R3_evaluated_child_count": int(phase3_child_count),
            }
        )
        return output

    scored_surface = row.get("scored_surface_records")
    if not isinstance(scored_surface, list):
        raise ValueError(
            f"history[{index}] lacks projected scored_surface_records."
        )
    scored_surface_size = _nonnegative_int(
        row.get("scored_surface_size"),
        field=f"history[{index}].scored_surface_size",
    )
    if scored_surface_size != len(scored_surface) or len(scored_surface) != r2:
        raise ValueError(
            f"history[{index}] projected Phase-I parent shortlist does not "
            f"close: retained_parents={r2}, scored_surface_size="
            f"{scored_surface_size}, records={len(scored_surface)}."
        )

    split_parent_count = 0
    split_child_count = 0
    unsplit_singleton_count = 0
    for record_index, record in enumerate(scored_surface):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"history[{index}].scored_surface_records[{record_index}] "
                "is not a mapping."
            )
        chosen = str(
            record.get("runtime_split_chosen_representation") or "parent"
        ).strip().lower()
        split_mode = str(
            record.get("runtime_split_mode") or "off"
        ).strip().lower()
        if chosen == "child_set":
            child_count = _nonnegative_int(
                record.get("runtime_split_child_count"),
                field=(
                    f"history[{index}].scored_surface_records[{record_index}]."
                    "runtime_split_child_count"
                ),
            )
            if child_count < 1 or split_mode != "shortlist_pauli_children_v1":
                raise ValueError(
                    f"history[{index}] split parent {record_index} has no "
                    "valid projected-singleton child population."
                )
            subset_lengths = []
            for field_name in (
                "runtime_split_child_indices",
                "runtime_split_child_labels",
                "runtime_split_child_generator_ids",
            ):
                values = record.get(field_name)
                if not isinstance(values, list):
                    raise ValueError(
                        f"history[{index}] split parent {record_index} lacks "
                        f"{field_name}."
                    )
                subset_lengths.append(len(values))
            if any(length != 1 for length in subset_lengths):
                raise ValueError(
                    f"history[{index}] split parent {record_index} is not the "
                    "Paper-I singleton-subset route."
                )
            split_parent_count += 1
            split_child_count += child_count
            continue
        if chosen == "parent" and split_mode == "off":
            child_count_value = record.get("runtime_split_child_count")
            if child_count_value not in {None, 0}:
                raise ValueError(
                    f"history[{index}] unsplit singleton {record_index} "
                    "declares split children."
                )
            unsplit_singleton_count += 1
            continue
        raise ValueError(
            f"history[{index}] has unsupported projected child-surface record "
            f"{record_index}: representation={chosen!r}, mode={split_mode!r}."
        )

    child_population = int(split_child_count + unsplit_singleton_count)
    if child_population < 1:
        raise ValueError(
            f"history[{index}] projected route produced no Phase-II children."
        )
    for field_name in (
        "method_input_candidate_count_total",
        "actual_evaluated_candidate_count_total",
        "candidate_count_total",
        "pre_shortlist_count_total",
        "records_evaluated",
    ):
        observed = _nonnegative_int(
            phase2_scope.get(field_name),
            field=(
                f"history[{index}].controller.by_scope."
                f"phase2_rerank_records.{field_name}"
            ),
        )
        # Historical Paper-I telemetry compacted all children generated from
        # one retained parent into one surfaced record.  Current clean-route
        # telemetry exposes every child directly.  Both must close to the
        # deterministic expanded child population used for S_alg.
        if observed not in {r2, child_population}:
            raise ValueError(
                f"history[{index}] projected Phase-II input does not close: "
                f"retained_parents={r2}, expanded_children="
                f"{child_population}, {field_name}={observed}."
            )
    if not isinstance(phase3, Mapping):
        raise ValueError(
            f"history[{index}] lacks the projected Phase-III child shortlist."
        )
    phase3_child_count = _nonnegative_int(
        phase3.get("method_input_candidate_count_total"),
        field=(
            f"history[{index}].controller.by_phase.phase3."
            "method_input_candidate_count_total"
        ),
    )
    phase3_scope = _controller_event_scope(
        controller,
        history_index=index,
        phase="phase3",
        event_kind="phase3_reduced_geometry_rerank",
    )
    for field_name in (
        "method_input_candidate_count_total",
        "candidate_count_total",
        "actual_evaluated_candidate_count_total",
        "pre_shortlist_count_total",
        "records_evaluated",
    ):
        observed = _nonnegative_int(
            phase3_scope.get(field_name),
            field=(
                f"history[{index}].controller.by_scope."
                f"phase3_reduced_geometry_rerank.{field_name}"
            ),
        )
        if observed != phase3_child_count:
            raise ValueError(
                f"history[{index}] projected Phase-III child shortlist does "
                f"not close: R3={phase3_child_count}, "
                f"{field_name}={observed}."
            )
    if phase3_child_count < 1 or phase3_child_count > child_population:
        raise ValueError(
            f"history[{index}] projected Phase-III child count is outside "
            f"the Phase-II child population: R2={child_population}, "
            f"R3={phase3_child_count}."
        )
    output.update(
        {
            "phase1_retained_parent_count": int(r2),
            "R2": int(child_population),
            "R3": int(phase3_child_count),
            "R2_split_parent_count": int(split_parent_count),
            "R2_split_child_count": int(split_child_count),
            "R2_unsplit_singleton_count": int(unsplit_singleton_count),
            "R2_evaluated_child_count": int(child_population),
            "R3_evaluated_child_count": int(phase3_child_count),
        }
    )
    return output


def snake_clean_prefix_work(
    *,
    history: Sequence[Mapping[str, Any]],
    accepted_prefix_length: int,
    representation: str,
    estimator_ledger_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Recount one no-beam/no-prune Paper-I SNAKE accepted prefix."""

    k = _nonnegative_int(
        accepted_prefix_length, field="accepted_prefix_length"
    )
    if k < 1 or k > len(history):
        raise ValueError(
            "accepted_prefix_length is outside the completed history."
        )
    representation_key = str(representation).strip().lower()
    if representation_key not in _SNAKE_REPRESENTATIONS:
        raise ValueError(
            f"representation must be one of {sorted(_SNAKE_REPRESENTATIONS)!r}."
        )
    raw = estimator_ledger_receipt.get("cumulative_raw_occurrences")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "SNAKE estimator receipt lacks cumulative raw occurrences."
        )
    raw_components = _component_mapping(
        raw, field="cumulative_raw_occurrences"
    )
    raw_total = _nonnegative_int(
        raw.get("total"), field="cumulative_raw_occurrences.total"
    )
    if sum(raw_components.values()) != raw_total:
        raise ValueError("SNAKE raw estimator receipt does not close.")
    raw_h_outer = int(raw_components["N_H_outer"])
    if raw_h_outer not in {k, k + 1}:
        raise ValueError(
            "SNAKE prefix does not have either the clean K or audited legacy "
            "K+1 initial/outer Hamiltonian pattern."
        )
    redundant_initial_outer_refresh = int(raw_h_outer - k)

    rounds = [
        snake_round_cardinality_from_history(
            row,
            history_index=index,
            representation=representation_key,
        )
        for index, row in enumerate(history[:k])
    ]
    round_refit_occurrences = [
        _nonnegative_int(
            row.get("nfev_opt"),
            field=f"history[{index}].nfev_opt",
        )
        for index, row in enumerate(history[:k])
    ]
    if sum(round_refit_occurrences) != raw_components["N_H_refit"]:
        raise ValueError(
            "SNAKE cumulative Powell-objective occurrences do not close to "
            "the per-round nfev_opt telemetry."
        )
    for round_row, nfev_opt in zip(
        rounds, round_refit_occurrences, strict=True
    ):
        round_row["N_H_refit"] = int(nfev_opt)
    n_grad = 0
    n_metric = 0
    for round_row in rounds:
        n = int(round_row["n_active"])
        r1 = int(round_row["R1"])
        r2 = int(round_row["R2"])
        r3 = int(round_row["R3"])
        if representation_key == SNAKE_REPRESENTATION_INTACT_MACRO:
            n_grad += r1 + n
            n_metric += r1 + r2 + n * (n + 1) + 2 * n * r3
        else:
            # Phase I measures parent gradients and self metrics.  After the
            # Phase-I parent shortlist, splitting creates the child population
            # that enters Phase II.  Each child then requires its own gradient,
            # self metric, and self Hessian.  The parent Phase-II Hessian probe
            # from the historical implementation was dead work and is not
            # part of the declared algorithm.  The stored Paper-I route
            # evaluated the complete child population through the joint
            # response only for the Phase-II child shortlist.
            n_grad += r1 + r2 + n
            n_metric += r1 + 2 * r2 + n * (n + 1) + 2 * n * r3

    components = {
        "N_H_outer": int(k),
        "N_H_refit": int(raw_components["N_H_refit"]),
        "N_grad": int(n_grad),
        "N_metric": int(n_metric),
    }
    for component, clean_count in components.items():
        if int(raw_components[component]) < int(clean_count):
            raise ValueError(
                "SNAKE runtime estimator receipt cannot support the clean "
                f"{component} recount: raw={raw_components[component]}, "
                f"clean={clean_count}."
            )
    return _work_receipt(
        method="SNAKE",
        representation=representation_key,
        accepted_prefix_length=k,
        components=components,
        normalization={
            "hamiltonian_objectives": (
                "raw Powell objective occurrences retained; one current-state "
                "Hamiltonian evaluation per accepted round"
            ),
            "redundant_initial_outer_refresh_count_removed": int(
                redundant_initial_outer_refresh
            ),
            "same_iteration_reuse": (
                (
                    "intact-macro candidate self quantities carry forward "
                    "from Phases I/II; the active-only Phase-III scaffold is "
                    "acquired once per round"
                )
                if representation_key == SNAKE_REPRESENTATION_INTACT_MACRO
                else (
                    "child self quantities acquired after the Phase-I split "
                    "are carried from Phase II into Phase III; the active-only "
                    "Phase-III scaffold is acquired once per round"
                )
            ),
            "projected_singleton": (
                "Phase-I parents are shortlisted before splitting; every "
                "generated child and unsplit singleton enters Phase II; "
                "dead parent Phase-II work is excluded"
            ),
            "runtime_occurrence_total_diagnostic": int(raw_total),
            "runtime_occurrence_components_diagnostic": dict(raw_components),
        },
        round_cardinalities=rounds,
    )


def runtime_prefix_work(
    *,
    method: str,
    representation: str,
    accepted_prefix_length: int,
    estimator_ledger_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Close ``S_alg`` directly from an executed-call prefix receipt.

    This is the canonical path for current routes whose runtime ledger already
    records every required logical scalar-estimator invocation.  Unlike the
    archived-history recount, it does not infer candidate work from set
    cardinalities, which is essential for insertion-position routes that
    deliberately reuse one acquired quantity across commuting positions.
    """

    k = _nonnegative_int(
        accepted_prefix_length, field="accepted_prefix_length"
    )
    if k < 1:
        raise ValueError("accepted_prefix_length must be positive.")
    if estimator_ledger_receipt.get("status") != "complete":
        raise ValueError("runtime estimator receipt is not complete.")
    receipt_iteration = _nonnegative_int(
        estimator_ledger_receipt.get("outer_iteration"),
        field="estimator_ledger_receipt.outer_iteration",
    )
    if receipt_iteration != k:
        raise ValueError(
            "runtime estimator receipt does not match the accepted prefix."
        )
    raw = estimator_ledger_receipt.get("cumulative_raw_occurrences")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "runtime estimator receipt lacks cumulative raw occurrences."
        )
    components = _component_mapping(
        raw, field="cumulative_raw_occurrences"
    )
    raw_total = _nonnegative_int(
        raw.get("total"), field="cumulative_raw_occurrences.total"
    )
    if sum(components.values()) != raw_total:
        raise ValueError("runtime estimator receipt does not close.")
    raw_h_outer = int(components["N_H_outer"])
    if raw_h_outer not in {k, k + 1}:
        raise ValueError(
            "runtime prefix does not have either the clean K or audited "
            "legacy K+1 initial/outer Hamiltonian pattern."
        )
    components["N_H_outer"] = int(k)
    return _work_receipt(
        method=method,
        representation=representation,
        accepted_prefix_length=k,
        components=components,
        normalization={
            "source": "closed_runtime_occurrence_ledger",
            "runtime_receipt_schema": estimator_ledger_receipt.get("schema"),
            "runtime_occurrence_total": int(raw_total),
            "redundant_initial_outer_refresh_count_removed": int(
                raw_h_outer - k
            ),
            "same_iteration_reuse": (
                "already-acquired quantities may be carried between phases "
                "or commuting insertion positions; every executed invocation "
                "remaining in the ledger is counted"
            ),
            "unique_primitive_diagnostic_excluded": True,
        },
    )


def append_clean_prefix_work(
    *,
    accepted_prefix_length: int,
    cumulative_occurrence_summary: Mapping[str, Any],
    redundant_post_refit_verification_count: int,
    representation: str,
) -> dict[str, Any]:
    """Recount one exact append-only Paper-I accepted prefix."""

    k = _nonnegative_int(
        accepted_prefix_length, field="accepted_prefix_length"
    )
    if k < 1:
        raise ValueError("accepted_prefix_length must be positive.")
    occurrence_components = cumulative_occurrence_summary.get(
        "component_occurrence_counts"
    )
    if not isinstance(occurrence_components, Mapping):
        raise ValueError(
            "Append cumulative occurrence summary lacks component counts."
        )
    raw_components = {
        component: _nonnegative_int(
            occurrence_components.get(component),
            field=(
                "cumulative_occurrence_summary."
                f"component_occurrence_counts.{component}"
            ),
        )
        for component in S_ALG_COMPONENTS
    }
    raw_total = _nonnegative_int(
        cumulative_occurrence_summary.get("total_call_occurrences"),
        field="cumulative_occurrence_summary.total_call_occurrences",
    )
    if sum(raw_components.values()) != raw_total:
        raise ValueError("Append raw estimator occurrence summary does not close.")
    if raw_components["N_H_outer"] != k:
        raise ValueError(
            "Append prefix outer-H occurrences do not equal its accepted "
            "iteration count."
        )
    if raw_components["N_metric"] != 0:
        raise ValueError("Append-only Paper-I route unexpectedly measured geometry.")
    redundant = _nonnegative_int(
        redundant_post_refit_verification_count,
        field="redundant_post_refit_verification_count",
    )
    if redundant > raw_components["N_H_refit"]:
        raise ValueError(
            "Redundant Append verifier count exceeds raw refit-H occurrences."
        )
    components = {
        "N_H_outer": int(k),
        "N_H_refit": int(raw_components["N_H_refit"] - redundant),
        "N_grad": int(raw_components["N_grad"]),
        "N_metric": 0,
    }
    return _work_receipt(
        method="Append-ADAPT",
        representation=str(representation),
        accepted_prefix_length=k,
        components=components,
        normalization={
            "hamiltonian_objectives": (
                "every Powell objective retained; redundant deterministic "
                "post-Powell endpoint verification removed"
            ),
            "redundant_post_refit_verification_count_removed": int(redundant),
            "runtime_occurrence_total_diagnostic": int(raw_total),
            "runtime_occurrence_components_diagnostic": dict(raw_components),
        },
    )


__all__ = [
    "PAPER_I_S_ALG_ACCOUNTING_SCHEMA",
    "PAPER_I_S_ALG_CONTRACT",
    "PROJECTED_PHASE_ORDER",
    "S_ALG_COMPONENTS",
    "SNAKE_REPRESENTATION_INTACT_MACRO",
    "SNAKE_REPRESENTATION_PROJECTED_SINGLETON",
    "append_clean_prefix_work",
    "runtime_prefix_work",
    "snake_clean_prefix_work",
    "snake_round_cardinality_from_history",
]
