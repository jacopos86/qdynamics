#!/usr/bin/env python3
"""Pure ragged reporting for the maximum-k50 Paper-I all-six campaign.

The campaign runner authenticates scientific and archive evidence before it
constructs a reporting-cell projection.  This module then validates that
immutable projection, preserves accepted and terminal attempts as different
domains, and derives comparisons without inventing accepted controller rounds.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


MODULE_PATH = Path(__file__).resolve()
REPO_ROOT = MODULE_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.adaptive_phase_contracts import (
    ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1,
)
from pipelines.static_adapt.ra_adapt.contracts import canonical_sha256


CAMPAIGN_ID = (
    "paper_i_ra_all6_adaptive_shortlist_append_then_plateau_maximum_k50_"
    "20260817_v1"
)
MAXIMUM_CONTROLLER_ROUNDS = 50
CELL_PROJECTION_SCHEMA = (
    "paper_i_ra_all6_maximum_k50_reporting_cell_projection_v1"
)
REPORT_SCHEMA = (
    "paper_i_ra_all6_adaptive_maximum_k50_ragged_comparison_v1"
)
CELL_COMPLETION_SCHEMA = (
    "paper_i_ra_all6_adaptive_maximum_k50_cell_completion_v1"
)
MAXIMUM_COMPLETION_KIND = "reached_maximum_controller_rounds_v1"
NATURAL_COMPLETION_KIND = (
    "authenticated_phase3_no_positive_natural_terminal_v1"
)

REGIMES = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)

IDENTITY_FIELDS = (
    "execution_id",
    "cell_ordinal",
    "block",
    "regime_id",
    "nph",
    "insertion_policy",
)
ACCEPTED_ROW_FIELDS = (
    *IDENTITY_FIELDS,
    "controller_round",
    "energy",
    "absolute_delta_e",
    "placement_state",
    "phase0_population_count",
    "phase0_retained_count",
    "phase_i_input_count",
    "phase_i_retained_count",
    "phase_ii_input_count",
    "phase_ii_retained_count",
    "phase_iii_input_count",
    "phase_iii_adaptive_retained_count",
    "phase_iii_final_singleton_count",
    "phase_iii_final_record_id",
    "selected_generator",
    "selected_operator",
    "selected_position",
    "s_alg",
    "n2q",
    "d2q",
    "dc",
    "checkpoint_sha256",
)
TERMINAL_ATTEMPT_FIELDS = (
    *IDENTITY_FIELDS,
    "attempted_controller_round",
    "terminal_controller_outcome",
    "placement_state",
    "phase0_population_count",
    "phase0_retained_count",
    "phase_i_input_count",
    "phase_i_retained_count",
    "phase_ii_input_count",
    "phase_ii_retained_count",
    "phase_iii_input_count",
    "phase_iii_adaptive_retained_count",
    "phase_iii_final_singleton_count",
    "terminal_phase3_selection_receipt_sha256",
    "terminal_active_prefix_checkpoint_sha256",
)
PAIR_METRICS = ("energy", "absolute_delta_e", "s_alg", "n2q", "d2q", "dc")
SHARED_PREFIX_PAIR_FIELDS = (
    "regime_id",
    "nph",
    "controller_round",
    "append_execution_id",
    "plateau_execution_id",
    "placement_activation_status",
    "selected_record_status",
    *(f"plateau_minus_append_{metric}" for metric in PAIR_METRICS),
)
REGIME_ENDPOINT_PAIR_FIELDS = (
    "regime_id",
    "nph",
    "append_execution_id",
    "plateau_execution_id",
    "append_accepted_controller_rounds",
    "plateau_accepted_controller_rounds",
    "placement_activation_status",
    "comparison_status",
    "null_reason",
    "controller_round",
    *(f"plateau_minus_append_{metric}" for metric in PAIR_METRICS),
)
CELL_OUTCOME_FIELDS = (
    *IDENTITY_FIELDS,
    "maximum_controller_rounds",
    "accepted_controller_rounds",
    "completion_kind",
    "terminal_attempted_controller_round",
    "terminal_controller_outcome",
    "summary_artifact_status",
    "placement_activation_status",
    "completion_sha256",
    "reporting_cell_projection_sha256",
    "archive_backed_closure_sha256",
)
REPORT_FIELDS = (
    "schema",
    "status",
    "campaign_id",
    "maximum_controller_rounds",
    "accepted_row_count",
    "terminal_attempt_count",
    "placement_factor_status",
    "accepted_rows",
    "terminal_attempts",
    "cell_outcomes",
    "shared_prefix_pairs",
    "regime_endpoint_pairs",
    "submission_authorized",
    "paper_adoption_authorized",
    "paper_evidence_adoption_authorized",
    "sha256",
)


class ReportingError(RuntimeError):
    """Fail-closed error for an invalid reporting projection."""


def _mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportingError(f"{owner} must be a mapping.")
    return copy.deepcopy(dict(value))


def _sequence(value: Any, *, owner: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ReportingError(f"{owner} must be a sequence.")
    return copy.deepcopy(list(value))


def _require_exact_fields(
    value: Mapping[str, Any], fields: Sequence[str], *, owner: str
) -> None:
    observed = set(value)
    expected = set(fields)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ReportingError(
            f"{owner} fields drifted (missing={missing}, extra={extra})."
        )


def _require_digest(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ReportingError(f"{owner} must be a lowercase SHA-256 digest.")
    return value


def _validate_signed(value: Any, *, owner: str) -> dict[str, Any]:
    payload = _mapping(value, owner=owner)
    observed = payload.pop("sha256", None)
    if observed != canonical_sha256(payload):
        raise ReportingError(f"{owner} digest drifted.")
    return {**payload, "sha256": str(observed)}


def _require_int(value: Any, *, owner: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReportingError(f"{owner} must be an integer >= {minimum}.")
    return value


def _require_finite(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportingError(f"{owner} must be finite numeric evidence.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ReportingError(f"{owner} must be finite numeric evidence.")
    return converted


def _expected_identity(block: str, regime_index: int) -> dict[str, Any]:
    regime_id, nph = REGIMES[regime_index]
    insertion_policy = (
        "append_only" if block == "append" else "plateau_commutation"
    )
    ordinal = regime_index + 1 + (0 if block == "append" else len(REGIMES))
    return {
        "execution_id": (
            "all_phase_adaptive_natural_terminal__"
            f"{regime_id}__nph{nph}__{insertion_policy}__maximum_k50"
        ),
        "cell_ordinal": ordinal,
        "block": block,
        "regime_id": regime_id,
        "nph": nph,
        "insertion_policy": insertion_policy,
    }


def _validate_identity(value: Any, *, owner: str) -> dict[str, Any]:
    identity = _mapping(value, owner=owner)
    _require_exact_fields(identity, IDENTITY_FIELDS, owner=owner)
    matches = [
        expected
        for block in ("append", "plateau")
        for index in range(len(REGIMES))
        for expected in (_expected_identity(block, index),)
        if identity == expected
    ]
    if len(matches) != 1:
        raise ReportingError(f"{owner} is not a canonical campaign cell.")
    return identity


def _validate_count_chain(row: Mapping[str, Any], *, owner: str) -> None:
    names = (
        "phase0_population_count",
        "phase0_retained_count",
        "phase_i_input_count",
        "phase_i_retained_count",
        "phase_ii_input_count",
        "phase_ii_retained_count",
        "phase_iii_input_count",
        "phase_iii_adaptive_retained_count",
        "phase_iii_final_singleton_count",
    )
    counts = {
        name: _require_int(row.get(name), owner=f"{owner}.{name}")
        for name in names
    }
    if not (
        counts["phase0_retained_count"]
        <= counts["phase0_population_count"]
        and counts["phase_i_input_count"]
        == counts["phase0_retained_count"]
        and counts["phase_i_retained_count"]
        <= counts["phase_i_input_count"]
        and counts["phase_ii_input_count"]
        == counts["phase_i_retained_count"]
        and counts["phase_ii_retained_count"]
        <= counts["phase_ii_input_count"]
        and counts["phase_iii_input_count"]
        == counts["phase_ii_retained_count"]
        and counts["phase_iii_adaptive_retained_count"]
        <= counts["phase_iii_input_count"]
        and counts["phase_iii_final_singleton_count"]
        <= counts["phase_iii_adaptive_retained_count"]
    ):
        raise ReportingError(f"{owner} adaptive phase cardinalities drifted.")


def _validate_placement_state(
    identity: Mapping[str, Any], value: Any, *, owner: str
) -> str:
    if identity["block"] == "append":
        if value != "append_only":
            raise ReportingError(f"{owner} append placement state drifted.")
    elif value not in {"open", "closed"}:
        raise ReportingError(f"{owner} plateau placement state drifted.")
    return str(value)


def _validate_accepted_row(
    value: Any,
    *,
    identity: Mapping[str, Any],
    controller_round: int,
) -> dict[str, Any]:
    row = _mapping(value, owner="accepted row")
    _require_exact_fields(row, ACCEPTED_ROW_FIELDS, owner="accepted row")
    if {field: row.get(field) for field in IDENTITY_FIELDS} != dict(identity):
        raise ReportingError("Accepted row cell identity drifted.")
    if row.get("controller_round") != controller_round:
        raise ReportingError("Accepted controller rounds are not contiguous.")
    _require_finite(row.get("energy"), owner="accepted row energy")
    absolute_delta_e = _require_finite(
        row.get("absolute_delta_e"), owner="accepted row absolute_delta_e"
    )
    if absolute_delta_e < 0:
        raise ReportingError("Accepted row absolute_delta_e must be nonnegative.")
    _validate_placement_state(
        identity, row.get("placement_state"), owner="accepted row"
    )
    _validate_count_chain(row, owner="accepted row")
    if row.get("phase_iii_final_singleton_count") != 1:
        raise ReportingError("Accepted row must retain one Phase-III singleton.")
    for field in (
        "phase_iii_final_record_id",
        "selected_generator",
        "selected_operator",
    ):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ReportingError(f"Accepted row {field} is absent.")
    for field in ("selected_position", "s_alg", "n2q", "d2q", "dc"):
        _require_int(row.get(field), owner=f"accepted row {field}")
    _require_digest(row.get("checkpoint_sha256"), owner="accepted checkpoint")
    return row


def _validate_terminal_attempt(
    value: Any,
    *,
    identity: Mapping[str, Any],
    attempted_round: int,
) -> dict[str, Any]:
    row = _mapping(value, owner="terminal attempt")
    _require_exact_fields(
        row, TERMINAL_ATTEMPT_FIELDS, owner="terminal attempt"
    )
    if {field: row.get(field) for field in IDENTITY_FIELDS} != dict(identity):
        raise ReportingError("Terminal-attempt cell identity drifted.")
    if row.get("attempted_controller_round") != attempted_round:
        raise ReportingError("Terminal attempted controller round drifted.")
    if (
        row.get("terminal_controller_outcome")
        != ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
    ):
        raise ReportingError("Terminal controller outcome drifted.")
    _validate_placement_state(
        identity, row.get("placement_state"), owner="terminal attempt"
    )
    _validate_count_chain(row, owner="terminal attempt")
    if row.get("phase_iii_final_singleton_count") != 0:
        raise ReportingError("Terminal attempt cannot retain a singleton.")
    _require_digest(
        row.get("terminal_phase3_selection_receipt_sha256"),
        owner="terminal Phase-III receipt",
    )
    _require_digest(
        row.get("terminal_active_prefix_checkpoint_sha256"),
        owner="terminal active-prefix checkpoint",
    )
    return row


def _validate_completion(
    value: Any, *, identity: Mapping[str, Any], accepted_rounds: int
) -> dict[str, Any]:
    completion = _validate_signed(value, owner="cell completion")
    required = {
        "schema",
        "campaign_id",
        "execution_id",
        "cell_ordinal",
        "completion_kind",
        "maximum_controller_rounds",
        "accepted_controller_rounds",
        "terminal_attempted_controller_round",
        "terminal_controller_outcome",
        "terminal_phase3_selection_receipt_sha256",
        "summary_artifact_status",
        "sha256",
    }
    if not required.issubset(completion):
        raise ReportingError("Cell completion fields are incomplete.")
    if (
        completion["schema"] != CELL_COMPLETION_SCHEMA
        or completion["campaign_id"] != CAMPAIGN_ID
        or completion["execution_id"] != identity["execution_id"]
        or completion["cell_ordinal"] != identity["cell_ordinal"]
        or completion["maximum_controller_rounds"]
        != MAXIMUM_CONTROLLER_ROUNDS
        or completion["accepted_controller_rounds"] != accepted_rounds
    ):
        raise ReportingError("Cell completion binding drifted.")
    return completion


def _validate_cell_projection(value: Any) -> dict[str, Any]:
    projection = _validate_signed(value, owner="reporting cell projection")
    _require_exact_fields(
        projection,
        (
            "schema",
            "cell",
            "completion",
            "accepted_rows",
            "terminal_attempt",
            "archive_backed_closure_sha256",
            "sha256",
        ),
        owner="reporting cell projection",
    )
    if projection["schema"] != CELL_PROJECTION_SCHEMA:
        raise ReportingError("Reporting cell projection schema drifted.")
    identity = _validate_identity(
        projection["cell"], owner="reporting cell identity"
    )
    raw_rows = _sequence(
        projection["accepted_rows"], owner="accepted reporting rows"
    )
    if len(raw_rows) > MAXIMUM_CONTROLLER_ROUNDS:
        raise ReportingError("Accepted rows exceed the maximum-k50 horizon.")
    rows = [
        _validate_accepted_row(
            row, identity=identity, controller_round=controller_round
        )
        for controller_round, row in enumerate(raw_rows, start=1)
    ]
    completion = _validate_completion(
        projection["completion"],
        identity=identity,
        accepted_rounds=len(rows),
    )
    kind = completion["completion_kind"]
    if kind == NATURAL_COMPLETION_KIND:
        if len(rows) >= MAXIMUM_CONTROLLER_ROUNDS:
            raise ReportingError("Natural terminal must occur before k50.")
        attempted_round = len(rows) + 1
        if (
            completion["terminal_attempted_controller_round"]
            != attempted_round
            or completion["terminal_controller_outcome"]
            != ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
            or completion["summary_artifact_status"]
            != ("not_applicable_round_zero" if not rows else "present")
        ):
            raise ReportingError("Natural-terminal completion drifted.")
        terminal = _validate_terminal_attempt(
            projection["terminal_attempt"],
            identity=identity,
            attempted_round=attempted_round,
        )
        if (
            terminal["terminal_phase3_selection_receipt_sha256"]
            != completion["terminal_phase3_selection_receipt_sha256"]
        ):
            raise ReportingError(
                "Terminal Phase-III receipt binding drifted."
            )
    elif kind == MAXIMUM_COMPLETION_KIND:
        if (
            len(rows) != MAXIMUM_CONTROLLER_ROUNDS
            or completion["terminal_attempted_controller_round"] is not None
            or completion["terminal_controller_outcome"] is not None
            or completion["terminal_phase3_selection_receipt_sha256"] is not None
            or completion["summary_artifact_status"] != "present"
            or projection["terminal_attempt"] is not None
        ):
            raise ReportingError("Maximum-k50 completion drifted.")
        terminal = None
    else:
        raise ReportingError("Unauthorized cell completion kind.")
    archive_sha256 = _require_digest(
        projection["archive_backed_closure_sha256"],
        owner="archive-backed cell closure",
    )
    return {
        "identity": identity,
        "completion": completion,
        "accepted_rows": rows,
        "terminal_attempt": terminal,
        "archive_backed_closure_sha256": archive_sha256,
        "projection_sha256": projection["sha256"],
    }


def _placement_activation_status(cell: Mapping[str, Any]) -> str:
    identity = cell["identity"]
    if identity["block"] == "append":
        return "append_only"
    if any(
        row["placement_state"] == "open" for row in cell["accepted_rows"]
    ):
        return "activated_with_accepted_transition"
    terminal = cell["terminal_attempt"]
    if terminal is not None and terminal["placement_state"] == "open":
        return "activated_terminal_attempt_only"
    return "not_activated"


def _metric_deltas(
    append_row: Mapping[str, Any], plateau_row: Mapping[str, Any]
) -> dict[str, int | float]:
    return {
        f"plateau_minus_append_{metric}": (
            plateau_row[metric] - append_row[metric]
        )
        for metric in PAIR_METRICS
    }


def _selected_record_status(
    append_row: Mapping[str, Any], plateau_row: Mapping[str, Any]
) -> str:
    fields = (
        "phase_iii_final_record_id",
        "selected_generator",
        "selected_operator",
        "selected_position",
    )
    return (
        "agree"
        if tuple(append_row[field] for field in fields)
        == tuple(plateau_row[field] for field in fields)
        else "diverge"
    )


def _build_pairs(
    ordered: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shared_prefix_pairs: list[dict[str, Any]] = []
    endpoint_pairs: list[dict[str, Any]] = []
    for regime_index, (regime_id, nph) in enumerate(REGIMES):
        append = ordered[regime_index]
        plateau = ordered[len(REGIMES) + regime_index]
        append_rows = append["accepted_rows"]
        plateau_rows = plateau["accepted_rows"]
        activation_status = _placement_activation_status(plateau)
        common_rounds = min(len(append_rows), len(plateau_rows))
        for round_index in range(common_rounds):
            append_row = append_rows[round_index]
            plateau_row = plateau_rows[round_index]
            pair = {
                "regime_id": regime_id,
                "nph": nph,
                "controller_round": round_index + 1,
                "append_execution_id": append["identity"]["execution_id"],
                "plateau_execution_id": plateau["identity"]["execution_id"],
                "placement_activation_status": activation_status,
                "selected_record_status": _selected_record_status(
                    append_row, plateau_row
                ),
                **_metric_deltas(append_row, plateau_row),
            }
            _require_exact_fields(
                pair, SHARED_PREFIX_PAIR_FIELDS, owner="shared-prefix pair"
            )
            shared_prefix_pairs.append(pair)

        append_count = len(append_rows)
        plateau_count = len(plateau_rows)
        endpoint = {
            "regime_id": regime_id,
            "nph": nph,
            "append_execution_id": append["identity"]["execution_id"],
            "plateau_execution_id": plateau["identity"]["execution_id"],
            "append_accepted_controller_rounds": append_count,
            "plateau_accepted_controller_rounds": plateau_count,
            "placement_activation_status": activation_status,
        }
        if append_count != plateau_count:
            endpoint.update(
                {
                    "comparison_status": "not_compared",
                    "null_reason": "accepted_controller_round_mismatch",
                    "controller_round": None,
                    **{
                        f"plateau_minus_append_{metric}": None
                        for metric in PAIR_METRICS
                    },
                }
            )
        elif append_count == 0:
            endpoint.update(
                {
                    "comparison_status": "not_compared",
                    "null_reason": "no_accepted_controller_rounds",
                    "controller_round": None,
                    **{
                        f"plateau_minus_append_{metric}": None
                        for metric in PAIR_METRICS
                    },
                }
            )
        else:
            endpoint.update(
                {
                    "comparison_status": "compared",
                    "null_reason": None,
                    "controller_round": append_count,
                    **_metric_deltas(append_rows[-1], plateau_rows[-1]),
                }
            )
        _require_exact_fields(
            endpoint, REGIME_ENDPOINT_PAIR_FIELDS, owner="regime endpoint pair"
        )
        endpoint_pairs.append(endpoint)
    return shared_prefix_pairs, endpoint_pairs


def build_ragged_report(
    cell_projections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate twelve cell projections and return canonical ragged evidence."""

    raw_cells = _sequence(
        cell_projections, owner="maximum-k50 reporting cells"
    )
    validated = [_validate_cell_projection(value) for value in raw_cells]
    by_execution = {
        cell["identity"]["execution_id"]: cell for cell in validated
    }
    expected_identities = [
        _expected_identity(block, regime_index)
        for block in ("append", "plateau")
        for regime_index in range(len(REGIMES))
    ]
    if (
        len(validated) != len(expected_identities)
        or len(by_execution) != len(expected_identities)
        or set(by_execution)
        != {identity["execution_id"] for identity in expected_identities}
    ):
        raise ReportingError("Reporting cells are not the exact all-six matrix.")
    ordered = [
        by_execution[identity["execution_id"]] for identity in expected_identities
    ]
    accepted_rows = [
        copy.deepcopy(row)
        for cell in ordered
        for row in cell["accepted_rows"]
    ]
    terminal_attempts = [
        copy.deepcopy(cell["terminal_attempt"])
        for cell in ordered
        if cell["terminal_attempt"] is not None
    ]
    cell_outcomes = [
        {
            **cell["identity"],
            "maximum_controller_rounds": MAXIMUM_CONTROLLER_ROUNDS,
            "accepted_controller_rounds": len(cell["accepted_rows"]),
            "completion_kind": cell["completion"]["completion_kind"],
            "terminal_attempted_controller_round": cell["completion"][
                "terminal_attempted_controller_round"
            ],
            "terminal_controller_outcome": cell["completion"][
                "terminal_controller_outcome"
            ],
            "summary_artifact_status": cell["completion"][
                "summary_artifact_status"
            ],
            "placement_activation_status": _placement_activation_status(cell),
            "completion_sha256": cell["completion"]["sha256"],
            "reporting_cell_projection_sha256": cell["projection_sha256"],
            "archive_backed_closure_sha256": cell[
                "archive_backed_closure_sha256"
            ],
        }
        for cell in ordered
    ]
    plateau_statuses = [
        row["placement_activation_status"]
        for row in cell_outcomes
        if row["block"] == "plateau"
    ]
    if "activated_with_accepted_transition" in plateau_statuses:
        placement_status = "activated_with_accepted_transition"
    elif "activated_terminal_attempt_only" in plateau_statuses:
        placement_status = "activated_terminal_attempt_only"
    else:
        placement_status = "not_activated"
    shared_prefix_pairs, endpoint_pairs = _build_pairs(ordered)
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "passed_all6_append_then_plateau_maximum_k50",
        "campaign_id": CAMPAIGN_ID,
        "maximum_controller_rounds": MAXIMUM_CONTROLLER_ROUNDS,
        "accepted_row_count": len(accepted_rows),
        "terminal_attempt_count": len(terminal_attempts),
        "placement_factor_status": placement_status,
        "accepted_rows": accepted_rows,
        "terminal_attempts": terminal_attempts,
        "cell_outcomes": cell_outcomes,
        "shared_prefix_pairs": shared_prefix_pairs,
        "regime_endpoint_pairs": endpoint_pairs,
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _validated_report(value: Any) -> dict[str, Any]:
    report = _validate_signed(value, owner="maximum-k50 ragged report")
    _require_exact_fields(report, REPORT_FIELDS, owner="maximum-k50 ragged report")
    if (
        report["schema"] != REPORT_SCHEMA
        or report["status"]
        != "passed_all6_append_then_plateau_maximum_k50"
        or report["campaign_id"] != CAMPAIGN_ID
        or report["maximum_controller_rounds"] != MAXIMUM_CONTROLLER_ROUNDS
        or report["submission_authorized"] is not False
        or report["paper_adoption_authorized"] is not False
        or report["paper_evidence_adoption_authorized"] is not False
    ):
        raise ReportingError("Maximum-k50 ragged report identity drifted.")
    collections = (
        ("accepted_rows", ACCEPTED_ROW_FIELDS),
        ("terminal_attempts", TERMINAL_ATTEMPT_FIELDS),
        ("cell_outcomes", CELL_OUTCOME_FIELDS),
        ("shared_prefix_pairs", SHARED_PREFIX_PAIR_FIELDS),
        ("regime_endpoint_pairs", REGIME_ENDPOINT_PAIR_FIELDS),
    )
    for name, fields in collections:
        rows = _sequence(report[name], owner=f"report {name}")
        for row in rows:
            mapped = _mapping(row, owner=f"report {name} row")
            _require_exact_fields(mapped, fields, owner=f"report {name} row")
    if (
        report["accepted_row_count"] != len(report["accepted_rows"])
        or report["terminal_attempt_count"] != len(report["terminal_attempts"])
        or len(report["cell_outcomes"]) != 12
        or len(report["regime_endpoint_pairs"]) != len(REGIMES)
    ):
        raise ReportingError("Maximum-k50 ragged report cardinality drifted.")
    return report


def render_report_json(report: Mapping[str, Any]) -> str:
    """Render deterministic human-readable JSON with one trailing LF."""

    validated = _validated_report(report)
    return (
        json.dumps(
            validated,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _render_csv(
    report: Mapping[str, Any], *, collection: str, fields: Sequence[str]
) -> str:
    validated = _validated_report(report)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(validated[collection])
    return stream.getvalue()


def render_accepted_rows_csv(report: Mapping[str, Any]) -> str:
    return _render_csv(
        report, collection="accepted_rows", fields=ACCEPTED_ROW_FIELDS
    )


def render_terminal_attempts_csv(report: Mapping[str, Any]) -> str:
    return _render_csv(
        report,
        collection="terminal_attempts",
        fields=TERMINAL_ATTEMPT_FIELDS,
    )


def render_cell_outcomes_csv(report: Mapping[str, Any]) -> str:
    return _render_csv(
        report, collection="cell_outcomes", fields=CELL_OUTCOME_FIELDS
    )


def render_shared_prefix_pairs_csv(report: Mapping[str, Any]) -> str:
    return _render_csv(
        report,
        collection="shared_prefix_pairs",
        fields=SHARED_PREFIX_PAIR_FIELDS,
    )


def render_regime_endpoint_pairs_csv(report: Mapping[str, Any]) -> str:
    return _render_csv(
        report,
        collection="regime_endpoint_pairs",
        fields=REGIME_ENDPOINT_PAIR_FIELDS,
    )


def render_csv_bundle(report: Mapping[str, Any]) -> dict[str, str]:
    """Render all fixed-schema CSV projections in stable filename order."""

    return {
        "accepted_rows.csv": render_accepted_rows_csv(report),
        "terminal_attempts.csv": render_terminal_attempts_csv(report),
        "cell_outcomes.csv": render_cell_outcomes_csv(report),
        "shared_prefix_pairs.csv": render_shared_prefix_pairs_csv(report),
        "regime_endpoint_pairs.csv": render_regime_endpoint_pairs_csv(report),
    }


def _markdown_text(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact deterministic diagnostic, including empty domains."""

    validated = _validated_report(report)
    lines = [
        "# Paper-I RA all-six maximum-k50 ragged diagnostic",
        "",
        "Diagnostic only; no manuscript or evidence adoption.",
        "",
        f"Placement factor: **{validated['placement_factor_status']}**.",
        f"Accepted rows: **{validated['accepted_row_count']}**.",
        f"Terminal attempts: **{validated['terminal_attempt_count']}**.",
        "",
        "## Accepted controller rounds",
        "",
    ]
    if not validated["accepted_rows"]:
        lines.extend(["No accepted controller rounds were recorded.", ""])
    else:
        lines.extend(
            [
                "| cell | k | E | |ΔE| | placement | selected | S_alg | "
                "N2q | D2q | Dc |",
                "|---|---:|---:|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in validated["accepted_rows"]:
            lines.append(
                f"| {_markdown_text(row['block'])}/{_markdown_text(row['regime_id'])} "
                f"| {row['controller_round']} | {row['energy']:.12g} "
                f"| {row['absolute_delta_e']:.4e} "
                f"| {_markdown_text(row['placement_state'])} "
                f"| {_markdown_text(row['selected_generator'])}@"
                f"{row['selected_position']} "
                f"| {row['s_alg']} | {row['n2q']} | {row['d2q']} | {row['dc']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Terminal attempts",
            "",
            "| cell | attempted k | outcome | placement | P0 | PI | PII | "
            "PIII adaptive/final |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in validated["terminal_attempts"]:
        lines.append(
            f"| {_markdown_text(row['block'])}/{_markdown_text(row['regime_id'])} "
            f"| {row['attempted_controller_round']} "
            f"| {_markdown_text(row['terminal_controller_outcome'])} "
            f"| {_markdown_text(row['placement_state'])} "
            f"| {row['phase0_population_count']}/{row['phase0_retained_count']} "
            f"| {row['phase_i_input_count']}/{row['phase_i_retained_count']} "
            f"| {row['phase_ii_input_count']}/{row['phase_ii_retained_count']} "
            f"| {row['phase_iii_adaptive_retained_count']}/"
            f"{row['phase_iii_final_singleton_count']} |"
        )
    lines.extend(
        [
            "",
            "## Regime endpoints",
            "",
            "| regime | append m | plateau m | placement | comparison | null "
            "reason | ΔE | Δ|E−E_exact| | ΔS_alg | ΔN2q | ΔD2q | ΔDc |",
            "|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in validated["regime_endpoint_pairs"]:
        lines.append(
            f"| {_markdown_text(row['regime_id'])} "
            f"| {row['append_accepted_controller_rounds']} "
            f"| {row['plateau_accepted_controller_rounds']} "
            f"| {_markdown_text(row['placement_activation_status'])} "
            f"| {_markdown_text(row['comparison_status'])} "
            f"| {_markdown_text(row['null_reason'])} "
            f"| {_markdown_text(row['plateau_minus_append_energy'])} "
            f"| {_markdown_text(row['plateau_minus_append_absolute_delta_e'])} "
            f"| {_markdown_text(row['plateau_minus_append_s_alg'])} "
            f"| {_markdown_text(row['plateau_minus_append_n2q'])} "
            f"| {_markdown_text(row['plateau_minus_append_d2q'])} "
            f"| {_markdown_text(row['plateau_minus_append_dc'])} |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "ACCEPTED_ROW_FIELDS",
    "CAMPAIGN_ID",
    "CELL_OUTCOME_FIELDS",
    "CELL_PROJECTION_SCHEMA",
    "MAXIMUM_CONTROLLER_ROUNDS",
    "REPORT_SCHEMA",
    "REGIMES",
    "REGIME_ENDPOINT_PAIR_FIELDS",
    "ReportingError",
    "SHARED_PREFIX_PAIR_FIELDS",
    "TERMINAL_ATTEMPT_FIELDS",
    "build_ragged_report",
    "render_accepted_rows_csv",
    "render_cell_outcomes_csv",
    "render_csv_bundle",
    "render_markdown",
    "render_regime_endpoint_pairs_csv",
    "render_report_json",
    "render_shared_prefix_pairs_csv",
    "render_terminal_attempts_csv",
]
