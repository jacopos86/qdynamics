#!/usr/bin/env python3
"""Results-first promotion contract for one SIAM exact/APM/AVQDS cell.

This module deliberately consumes only immutable offline-score documents,
Qiskit-free terminal-resource sidecars, and positive source-locked completion
receipts.  It never opens a raw trajectory or re-selects a run.  Completed but
out-of-gate arms remain numerical candidates; only an arm with no verified
completed score is represented as pending.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from pipelines.resource_estimation.pauli_graph_span import (
    PAPER_I_MARRAKESH_GRAPH_SPAN_PROFILE_V1,
    marrakesh_logical_embedding_v1,
)
from pipelines.time_dynamics.diagnostics.siam_offline_scoring import (
    SIAM_ACCURACY_POLICY_V2,
    SIAM_OFFLINE_SCORE_V3,
    _verified_score,
    classify_siam_matched_pair,
)
from pipelines.time_dynamics.diagnostics.siam_exact_reference_store import (
    SIAM_EXACT_PREFIX_VIEW_V2,
    SIAM_EXACT_REFERENCE_STORE_V2,
    SIAM_EXACT_SLICE_VIEW_V2,
    SIAMExactReferenceStore,
)
from pipelines.time_dynamics.diagnostics.siam_terminal_resources import (
    FAKE_MARRAKESH_BACKEND_NAME,
    FAKE_MARRAKESH_GRAPH_SNAPSHOT_V1,
    FAKE_MARRAKESH_NUM_QUBITS,
    FAKE_MARRAKESH_UNDIRECTED_EDGES_SHA256_V1,
    FAKE_MARRAKESH_UNDIRECTED_EDGES_V1,
    SIAM_TERMINAL_RESOURCE_V1,
)
from pipelines.time_dynamics.diagnostics.siam_segment_stitching import (
    SIAM_NB4_POINT_COUNT,
    SIAM_NB4_STITCH_COMPLETION_V1,
    SIAM_NB4_T_FINAL,
    validate_siam_nb4_stitch_completion_receipt,
)
from pipelines.time_dynamics.paper_ii_runs import (
    SIAM_NB4_CONTINUATION_SEGMENT,
    SIAM_NB4_CONTINUATION_TIMES,
    SIAM_NB4_FULL_TIMES,
    SIAM_NB4_SCREEN_SEGMENT,
    SIAM_NB4_SCREEN_TIMES,
)
from pipelines.time_dynamics.siam_chtc_package import (
    SIAM_NB4_CHTC_RECEIPT_SCHEMA_V3,
    SIAM_NB4_CHTC_RECEIPT_SCHEMA_V4,
    validate_siam_nb4_completion_receipt,
)
from src.quantum.siam_observables import SIAM_OBSERVABLES_V2
from src.quantum.siam_reference import SIAM_REFERENCE_V2


SIAM_RESULTS_PROMOTION_V1 = "siam_results_promotion_v1"
SIAM_RESULTS_PROMOTION_V2 = "siam_results_promotion_v2"
SIAM_RESULTS_LATEX_FRAGMENT_V1 = "siam_results_latex_fragment_v1"
SIAM_RESULTS_LATEX_FRAGMENT_V2 = "siam_results_latex_fragment_v2"
SIAM_RESULTS_PDF_V1 = "siam_results_pdf_v1"
SIAM_RESULTS_PDF_V2 = "siam_results_pdf_v2"
SIAM_RESULTS_PDF_RECEIPT_V1 = "siam_results_pdf_receipt_v1"
SIAM_RESULTS_PDF_RECEIPT_V2 = "siam_results_pdf_receipt_v2"
SIAM_RESULTS_DIAGNOSTIC_ROLE = "diagnostic_screening"
SIAM_RESULTS_MANUSCRIPT_ROLE = "manuscript_table"
_METHODS = ("avqds", "full_apm")
_STRUCTURAL_POLICY = {"avqds": "avqds", "full_apm": "exchange"}
_READER_LABEL = {"avqds": "AVQDS", "full_apm": "Full APM"}
_NB4_NUM_QUBITS = 10
_GRAPH_COMPARISON_SCHEMA = "pauli_graph_span_trajectory_comparison_v1"
_GRAPH_ABSOLUTE_SCHEMA = "pauli_graph_span_absolute_cost_v1"
_GRAPH_REPORTING_ROLE = "routing_independent_structural_cost_primary_v1"
_QISKIT_FREE_DECLARATION = {
    "required": False,
    "performed": False,
    "role": "optional_diagnostic_sidecar_only",
}
_NB4_PRIMITIVE_NAMES = ("m_d", "D_d", "C_1", "C_2", "C_3", "C_4")
_NB4_DERIVED_NAMES = ("Sigma_1", "Sigma_2", "Sigma_3", "Sigma_4")
_STITCHED_NB4_SEGMENT = "stitched_t0_t5_501"


class SIAMResultsContractError(ValueError):
    """A scored SIAM result cannot be promoted without ambiguity."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _document_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(dict(value))).hexdigest()


def _time_grid_sha256(times: Sequence[float]) -> str:
    return hashlib.sha256(
        _canonical_bytes([float(value) for value in times])
    ).hexdigest()


def _read_json_artifact(
    path: Path, *, label: str
) -> tuple[dict[str, Any], str]:
    try:
        content = path.read_bytes()
        payload = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SIAMResultsContractError(
            f"{label} is not readable JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise SIAMResultsContractError(f"{label} must be a JSON object: {path}")
    return payload, hashlib.sha256(content).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SIAMResultsContractError(f"{label} must be a JSON object.")
    return value


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SIAMResultsContractError(f"{label} must be numeric.") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise SIAMResultsContractError(f"{label} must be {qualifier}.")
    return result


def _integer(value: Any, *, label: str, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SIAMResultsContractError(f"{label} must be an integer.")
    result = int(value)
    if nonnegative and result < 0:
        raise SIAMResultsContractError(f"{label} must be nonnegative.")
    return result


def _sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise SIAMResultsContractError(f"{label} must be a SHA-256 digest.")
    return digest


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise SIAMResultsContractError(f"{label} must be Boolean.")
    return value


def _expected_graph_snapshot() -> dict[str, Any]:
    return {
        "schema": FAKE_MARRAKESH_GRAPH_SNAPSHOT_V1,
        "backend_name": FAKE_MARRAKESH_BACKEND_NAME,
        "num_qubits": FAKE_MARRAKESH_NUM_QUBITS,
        "undirected_coupling_edge_count": len(
            FAKE_MARRAKESH_UNDIRECTED_EDGES_V1
        ),
        "undirected_coupling_edges_sha256": (
            FAKE_MARRAKESH_UNDIRECTED_EDGES_SHA256_V1
        ),
        "source": "frozen_fake_marrakesh_conf_marrakesh_v1",
    }


def _canonical_graph_tuple(
    value: Mapping[str, Any],
    *,
    label: str,
    allow_negative: bool = False,
) -> dict[str, float | int]:
    expected_keys = {"C_hat_2q", "C_hat_d", "C_hat_1q", "N_theta"}
    if set(value) != expected_keys:
        raise SIAMResultsContractError(f"{label} has a noncanonical field set.")
    costs = {
        name: _finite(
            value.get(name),
            label=f"{label} {name}",
            nonnegative=not allow_negative,
        )
        for name in ("C_hat_2q", "C_hat_d", "C_hat_1q")
    }
    theta_value = _finite(
        value.get("N_theta"),
        label=f"{label} N_theta",
        nonnegative=not allow_negative,
    )
    if not theta_value.is_integer():
        raise SIAMResultsContractError(f"{label} N_theta must be integer-valued.")
    return {**costs, "N_theta": int(theta_value)}


def _graph_tuple_from_absolute(
    value: Mapping[str, Any],
    *,
    label: str,
    allow_negative: bool = False,
) -> dict[str, float | int]:
    return _canonical_graph_tuple(
        {
            "C_hat_2q": value.get("absolute_c_hat_2q"),
            "C_hat_d": value.get("absolute_c_hat_d"),
            "C_hat_1q": value.get("absolute_c_hat_1q"),
            "N_theta": value.get("absolute_theta_count"),
        },
        label=label,
        allow_negative=allow_negative,
    )


def _verified_absolute_graph_record(
    value: Any,
    *,
    slot: str,
    stage: str,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    record = dict(
        _mapping(value, label=f"{slot} graph-span {stage} record")
    )
    required_equal = {
        "schema": _GRAPH_ABSOLUTE_SCHEMA,
        "profile_id": PAPER_I_MARRAKESH_GRAPH_SPAN_PROFILE_V1,
        "backend_name": FAKE_MARRAKESH_BACKEND_NAME,
        "no_transpile": True,
        "routing_seed_used": False,
        "num_qubits": _NB4_NUM_QUBITS,
        "embedding_logical_to_physical": list(
            marrakesh_logical_embedding_v1(_NB4_NUM_QUBITS)
        ),
        "coupling_edge_count": len(FAKE_MARRAKESH_UNDIRECTED_EDGES_V1),
        "graph_snapshot": _expected_graph_snapshot(),
        "coefficient_tolerance": 1.0e-12,
    }
    for field, expected in required_equal.items():
        if record.get(field) != expected:
            raise SIAMResultsContractError(
                f"{slot} graph-span {stage} {field} is not canonical."
            )
    for field in (
        "ordered_generator_occurrence_count",
        "nonidentity_pauli_rotation_count",
        "independent_parameter_count",
    ):
        _integer(
            record.get(field),
            label=f"{slot} graph-span {stage} {field}",
            nonnegative=True,
        )
    support_sha256 = record.get("ordered_support_sha256")
    if not isinstance(support_sha256, str) or len(support_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in support_sha256
    ):
        raise SIAMResultsContractError(
            f"{slot} graph-span {stage} support digest is invalid."
        )
    resource_tuple = _graph_tuple_from_absolute(
        record, label=f"{slot} graph-span {stage} tuple"
    )
    if resource_tuple["N_theta"] != record["independent_parameter_count"]:
        raise SIAMResultsContractError(
            f"{slot} graph-span {stage} N_theta is internally inconsistent."
        )
    return record, resource_tuple


def _verified_response_metric_set(
    value: Any,
    *,
    expected_names: Sequence[str],
    label: str,
    response_floor: float,
) -> dict[str, dict[str, float]]:
    """Validate one complete, canonically named response-metric family."""

    metrics = _mapping(value, label=label)
    if set(metrics) != set(expected_names):
        raise SIAMResultsContractError(
            f"{label} must contain exactly {', '.join(expected_names)}."
        )
    verified: dict[str, dict[str, float]] = {}
    for name in expected_names:
        metric = _mapping(metrics[name], label=f"{label} {name}")
        if set(metric) != {
            "response_scale",
            "mean_absolute_error",
            "response_normalized_mean_absolute_error",
        }:
            raise SIAMResultsContractError(
                f"{label} {name} has a noncanonical field set."
            )
        scale = _finite(
            metric["response_scale"],
            label=f"{label} {name} response scale",
            nonnegative=True,
        )
        mean_absolute_error = _finite(
            metric["mean_absolute_error"],
            label=f"{label} {name} mean absolute error",
            nonnegative=True,
        )
        normalized_error = _finite(
            metric["response_normalized_mean_absolute_error"],
            label=f"{label} {name} normalized error",
            nonnegative=True,
        )
        expected_normalized = mean_absolute_error / max(scale, response_floor)
        if not math.isclose(
            normalized_error,
            expected_normalized,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise SIAMResultsContractError(
                f"{label} {name} normalization is inconsistent."
            )
        verified[name] = {
            "response_scale": scale,
            "mean_absolute_error": mean_absolute_error,
            "response_normalized_mean_absolute_error": normalized_error,
        }
    return verified


def _verified_exact_view_receipt(
    *,
    reference: Mapping[str, Any],
    score_source: Mapping[str, Any],
    grid: Mapping[str, Any],
    grid_times: tuple[float, ...],
    view_field: str,
    view_schema: str,
) -> dict[str, Any]:
    """Reopen one segmented exact view and bind it to its full source.

    A digest copied into a score is not itself evidence that the referenced
    receipt exists or that the receipt can still be regenerated from the
    immutable full exact artifact.  Promotion therefore verifies both layers:
    the receipt bytes and its canonical document digest, followed by the exact
    store's manifest/NPZ/array and view-regeneration checks.
    """

    score_view = _mapping(
        reference.get(view_field),
        label=f"SIAM exact-reference {view_field}",
    )
    receipt_path_value = score_view.get("receipt_path")
    if not isinstance(receipt_path_value, str) or not receipt_path_value:
        raise SIAMResultsContractError(
            "SIAM segmented score has no exact-reference view receipt path."
        )
    receipt_path = Path(receipt_path_value)
    receipt, file_sha256 = _read_json_artifact(
        receipt_path,
        label="SIAM exact-reference view receipt",
    )
    receipt_sha256 = str(receipt.get("receipt_sha256") or "")
    if (
        receipt.get("schema") != view_schema
        or score_view.get("schema") != view_schema
        or score_view.get("receipt_sha256") != receipt_sha256
        or receipt_sha256 != _document_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
    ):
        raise SIAMResultsContractError(
            "SIAM exact-reference view receipt digest/schema does not match "
            "the score."
        )

    receipt_source = _mapping(
        receipt.get("source"), label="SIAM exact-reference receipt source"
    )
    reference_key = str(receipt_source.get("reference_key") or "")
    manifest_path = Path(str(receipt_source.get("manifest_path") or ""))
    data_path = Path(str(receipt_source.get("data_path") or ""))
    if (
        manifest_path.name != "manifest.json"
        or data_path.name != "reference.npz"
        or manifest_path.parent != data_path.parent
        or manifest_path.parent.name != reference_key
    ):
        raise SIAMResultsContractError(
            "SIAM exact-reference view receipt has a noncanonical store layout."
        )
    store = SIAMExactReferenceStore(manifest_path.parent.parent)
    try:
        if view_field == "prefix_view":
            loaded_view = store.load_prefix_view(receipt_path)
            view_kind = "prefix"
        elif view_field == "slice_view":
            loaded_view = store.load_slice_view(receipt_path)
            view_kind = "slice"
        else:  # pragma: no cover - private caller invariant
            raise ValueError(f"unsupported SIAM exact-view field {view_field!r}")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SIAMResultsContractError(
            "SIAM exact-reference view receipt fails exact-reference-store "
            f"verification: {exc}"
        ) from exc

    full = loaded_view.source
    score_manifest_path = Path(str(reference.get("manifest_path") or ""))
    score_data_path = Path(str(reference.get("data_path") or ""))
    expected_full_identity = {
        "reference_key": str(reference.get("key") or ""),
        "manifest_sha256": str(reference.get("manifest_sha256") or ""),
        "data_sha256": str(reference.get("data_sha256") or ""),
    }
    observed_full_identity = {
        "reference_key": reference_key,
        "manifest_sha256": str(receipt_source.get("manifest_sha256") or ""),
        "data_sha256": str(receipt_source.get("data_sha256") or ""),
    }
    if (
        observed_full_identity != expected_full_identity
        or full.key != reference_key
        or full.manifest_sha256 != expected_full_identity["manifest_sha256"]
        or full.data_sha256 != expected_full_identity["data_sha256"]
        or score_manifest_path.resolve() != full.manifest_path.resolve()
        or score_data_path.resolve() != full.data_path.resolve()
        or manifest_path.resolve() != full.manifest_path.resolve()
        or data_path.resolve() != full.data_path.resolve()
        or reference.get("propagator_schema")
        != full.manifest.get("propagator_schema")
        or score_source.get("seed_sha256") != full.manifest.get("seed_sha256")
    ):
        raise SIAMResultsContractError(
            "SIAM exact-reference view does not bind the score's full "
            "reference identity."
        )

    full_times = tuple(float(value) for value in full.times)
    expected_full_times = tuple(float(value) for value in SIAM_NB4_FULL_TIMES)
    full_grid_sha256 = _time_grid_sha256(full_times)
    if (
        full_times != expected_full_times
        or receipt_source.get("full_point_count") != len(expected_full_times)
        or receipt_source.get("full_grid_sha256") != full_grid_sha256
        or full_grid_sha256 != _time_grid_sha256(expected_full_times)
    ):
        raise SIAMResultsContractError(
            "SIAM exact-reference view has an invalid full-grid identity."
        )

    view_contract = _mapping(
        receipt.get(view_kind),
        label=f"SIAM exact-reference {view_kind} contract",
    )
    receipt_times = tuple(float(value) for value in view_contract.get("times") or ())
    grid_sha256 = _time_grid_sha256(grid_times)
    if (
        receipt_times != grid_times
        or view_contract.get("point_count") != len(grid_times)
        or score_view.get("point_count") != len(grid_times)
        or grid.get("point_count") != len(grid_times)
        or view_contract.get("ordered_values_sha256") != grid_sha256
        or score_view.get("ordered_values_sha256") != grid_sha256
        or grid.get("ordered_values_sha256") != grid_sha256
        or not math.isclose(
            _finite(
                view_contract.get("t_initial"),
                label=f"SIAM exact-reference {view_kind} initial time",
            ),
            grid_times[0],
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            _finite(
                view_contract.get("t_final"),
                label=f"SIAM exact-reference {view_kind} terminal time",
            ),
            grid_times[-1],
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise SIAMResultsContractError(
            "SIAM exact-reference view bounds/grid do not match the score."
        )

    if view_kind == "prefix":
        start_index = 0
        stop_index_exclusive = len(grid_times)
        if grid_times != full_times[:stop_index_exclusive]:
            raise SIAMResultsContractError(
                "SIAM exact-reference prefix does not match the full-grid prefix."
            )
    else:
        try:
            start_index = full_times.index(grid_times[0])
        except ValueError as exc:  # pragma: no cover - store normally catches this
            raise SIAMResultsContractError(
                "SIAM exact-reference slice start is absent from the full grid."
            ) from exc
        stop_index_exclusive = start_index + len(grid_times)
        if (
            view_contract.get("start_index") != start_index
            or view_contract.get("stop_index_exclusive")
            != stop_index_exclusive
            or grid_times != full_times[start_index:stop_index_exclusive]
        ):
            raise SIAMResultsContractError(
                "SIAM exact-reference slice indices/bounds are invalid."
            )

    return {
        "schema": view_schema,
        "path": str(receipt_path.resolve()),
        "file_sha256": file_sha256,
        "receipt_sha256": receipt_sha256,
        "view_kind": view_kind,
        "reference_key": reference_key,
        "point_count": len(grid_times),
        "initial_time": grid_times[0],
        "terminal_time": grid_times[-1],
        "start_index": start_index,
        "stop_index_exclusive": stop_index_exclusive,
        "ordered_values_sha256": grid_sha256,
    }


def _verified_score_artifact(
    path: Path,
    *,
    expected_slot: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    document, file_sha256 = _read_json_artifact(
        path, label=f"{expected_slot} offline score"
    )
    try:
        verified = _verified_score(document)
    except (KeyError, TypeError, ValueError) as exc:
        raise SIAMResultsContractError(
            f"{expected_slot} offline score fails verification: {path}: {exc}"
        ) from exc
    if verified.get("schema") != SIAM_OFFLINE_SCORE_V3:
        raise SIAMResultsContractError(
            "SIAM promotion requires Sigma-v2 offline scores."
        )
    if verified.get("reporting_only") is not True:
        raise SIAMResultsContractError(
            "SIAM promotion requires an offline reporting-only score."
        )
    policy = _mapping(
        verified.get("accuracy_policy"), label="SIAM score accuracy policy"
    )
    if policy.get("schema") != SIAM_ACCURACY_POLICY_V2:
        raise SIAMResultsContractError("SIAM promotion policy identity mismatch.")
    thresholds = _mapping(
        verified.get("thresholds"), label="SIAM score thresholds"
    )
    response_floor = _finite(
        thresholds.get("response_floor"),
        label="SIAM score response floor",
        nonnegative=True,
    )
    if response_floor <= 0.0:
        raise SIAMResultsContractError("SIAM score response floor must be positive.")
    _verified_response_metric_set(
        verified.get("primitive_response_metrics"),
        expected_names=_NB4_PRIMITIVE_NAMES,
        label="SIAM Nb4 primitive response metrics",
        response_floor=response_floor,
    )
    _verified_response_metric_set(
        verified.get("derived_screening_metrics"),
        expected_names=_NB4_DERIVED_NAMES,
        label="SIAM Nb4 derived screening metrics",
        response_floor=response_floor,
    )
    method = _mapping(verified.get("method"), label="SIAM score method")
    expected_policy = _STRUCTURAL_POLICY[expected_slot]
    if method.get("structural_policy") != expected_policy:
        raise SIAMResultsContractError(
            f"{expected_slot} requires structural_policy={expected_policy!r}."
        )
    grid = _mapping(verified.get("grid"), label="SIAM Nb4 score grid")
    try:
        grid_times = tuple(float(value) for value in grid.get("times") or ())
    except (TypeError, ValueError) as exc:
        raise SIAMResultsContractError(
            "SIAM Nb4 score has a malformed ordered grid."
        ) from exc
    if (
        grid.get("point_count") != len(grid_times)
        or grid.get("ordered_values_sha256") != _time_grid_sha256(grid_times)
    ):
        raise SIAMResultsContractError(
            "SIAM Nb4 score point count or ordered-grid digest is invalid."
        )
    reference = _mapping(
        verified.get("reference"), label="SIAM Nb4 exact reference"
    )
    score_source = _mapping(verified.get("source"), label="SIAM score source")
    expected_view_field: str | None
    expected_view_schema: str | None
    if grid_times == tuple(SIAM_NB4_SCREEN_TIMES):
        expected_view_field = "prefix_view"
        expected_view_schema = SIAM_EXACT_PREFIX_VIEW_V2
    elif grid_times == tuple(SIAM_NB4_CONTINUATION_TIMES):
        expected_view_field = "slice_view"
        expected_view_schema = SIAM_EXACT_SLICE_VIEW_V2
    elif grid_times == tuple(SIAM_NB4_FULL_TIMES):
        expected_view_field = None
        expected_view_schema = None
    else:
        raise SIAMResultsContractError(
            "SIAM Nb4 promotion requires a registered screen, continuation, "
            "or full-T5 score grid."
        )
    present_views = {
        field for field in ("prefix_view", "slice_view") if field in reference
    }
    exact_view_artifact: dict[str, Any] | None = None
    if expected_view_field is None:
        if present_views:
            raise SIAMResultsContractError(
                "A full-T5 SIAM score must bind the full exact artifact directly."
            )
    else:
        if present_views != {expected_view_field}:
            raise SIAMResultsContractError(
                "A segmented SIAM score must bind the canonical exact-reference "
                f"{expected_view_field.replace('_', ' ')}."
            )
        assert expected_view_schema is not None
        exact_view_artifact = _verified_exact_view_receipt(
            reference=reference,
            score_source=score_source,
            grid=grid,
            grid_times=grid_times,
            view_field=expected_view_field,
            view_schema=expected_view_schema,
        )
    score_artifact: dict[str, Any] = {
        "path": str(path.resolve()),
        "file_sha256": file_sha256,
        "score_sha256": str(document["score_sha256"]),
    }
    if exact_view_artifact is not None:
        score_artifact["exact_view_receipt"] = exact_view_artifact
    return verified, score_artifact, document


def _verified_resource_artifact(
    path: Path,
    *,
    slot: str,
    score: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    document, file_sha256 = _read_json_artifact(
        path, label=f"{slot} terminal resource"
    )
    core = dict(document)
    digest = core.pop("resource_sha256", None)
    if digest != _document_sha256(core):
        raise SIAMResultsContractError(
            f"{slot} terminal-resource digest mismatch: {path}"
        )
    if core.get("schema") != SIAM_TERMINAL_RESOURCE_V1:
        raise SIAMResultsContractError(
            f"{slot} has an unsupported terminal-resource schema."
        )
    if core.get("problem_family") != "siam":
        raise SIAMResultsContractError(f"{slot} resource is not a SIAM record.")
    if core.get("parameterization_mode") != "logical_shared":
        raise SIAMResultsContractError(
            f"{slot} resource does not use logical-shared coordinates."
        )
    source = _mapping(score.get("source"), label=f"{slot} score source")
    if core.get("trajectory_json_sha256") != source.get("run_sha256"):
        raise SIAMResultsContractError(
            f"{slot} score and terminal resource bind different trajectories."
        )
    if core.get("seed_json_sha256") != source.get("seed_sha256"):
        raise SIAMResultsContractError(
            f"{slot} score and terminal resource bind different seeds."
        )
    qiskit = _mapping(
        core.get("qiskit_compilation"), label=f"{slot} Qiskit role"
    )
    if dict(qiskit) != _QISKIT_FREE_DECLARATION:
        raise SIAMResultsContractError(
            f"{slot} canonical resource record is not strictly Qiskit-free."
        )

    comparison = dict(
        _mapping(
            core.get("graph_span_comparison"),
            label=f"{slot} graph-span comparison",
        )
    )
    comparison_invariants = {
        "schema": _GRAPH_COMPARISON_SCHEMA,
        "profile_id": PAPER_I_MARRAKESH_GRAPH_SPAN_PROFILE_V1,
        "reporting_role": _GRAPH_REPORTING_ROLE,
        "no_transpile": True,
        "same_declared_graph_for_initial_and_terminal": True,
    }
    for field, expected in comparison_invariants.items():
        if comparison.get(field) != expected:
            raise SIAMResultsContractError(
                f"{slot} graph-span comparison {field} is not canonical."
            )
    initial_record, initial_tuple = _verified_absolute_graph_record(
        comparison.get("initial"), slot=slot, stage="initial"
    )
    terminal_record, terminal_tuple = _verified_absolute_graph_record(
        comparison.get("terminal"), slot=slot, stage="terminal"
    )
    delta_record = _mapping(
        comparison.get("terminal_minus_initial"),
        label=f"{slot} graph-span delta record",
    )
    expected_delta_fields = {
        "absolute_c_hat_2q",
        "absolute_c_hat_d",
        "absolute_c_hat_1q",
        "absolute_theta_count",
    }
    if set(delta_record) != expected_delta_fields:
        raise SIAMResultsContractError(
            f"{slot} graph-span delta has a noncanonical field set."
        )
    delta_tuple = _graph_tuple_from_absolute(
        delta_record,
        label=f"{slot} graph-span delta tuple",
        allow_negative=True,
    )
    calculated_delta: dict[str, float | int] = {
        name: float(terminal_tuple[name]) - float(initial_tuple[name])
        for name in ("C_hat_2q", "C_hat_d", "C_hat_1q")
    }
    calculated_delta["N_theta"] = int(terminal_tuple["N_theta"]) - int(
        initial_tuple["N_theta"]
    )
    if delta_tuple != calculated_delta:
        raise SIAMResultsContractError(
            f"{slot} graph-span delta differs from terminal minus initial."
        )

    top_level_tuples = {
        "initial_resource_tuple": initial_tuple,
        "terminal_resource_tuple": terminal_tuple,
        "terminal_minus_initial_resource_tuple": delta_tuple,
    }
    for field, expected in top_level_tuples.items():
        observed = _canonical_graph_tuple(
            _mapping(core.get(field), label=f"{slot} {field}"),
            label=f"{slot} {field}",
            allow_negative=field == "terminal_minus_initial_resource_tuple",
        )
        if observed != expected:
            raise SIAMResultsContractError(
                f"{slot} {field} differs from its graph-span derivation."
            )
    if terminal_record["ordered_generator_occurrence_count"] != _integer(
        core.get("terminal_logical_macro_count"),
        label=f"{slot} terminal logical macro count",
        nonnegative=True,
    ):
        raise SIAMResultsContractError(
            f"{slot} terminal graph occurrence count differs from its macros."
        )
    if terminal_record["nonidentity_pauli_rotation_count"] != _integer(
        core.get("terminal_pauli_rotation_count"),
        label=f"{slot} terminal Pauli rotation count",
        nonnegative=True,
    ):
        raise SIAMResultsContractError(
            f"{slot} terminal graph rotation count differs from its sidecar."
        )
    if initial_record["graph_snapshot"] != terminal_record["graph_snapshot"]:
        raise SIAMResultsContractError(
            f"{slot} initial and terminal graph snapshots differ."
        )
    return core, {
        "path": str(path.resolve()),
        "file_sha256": file_sha256,
        "resource_sha256": str(digest),
    }


def _verified_completion_artifact(
    path: Path,
    *,
    slot: str,
    score: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the source-locked worker's positive terminal receipt.

    Offline numerical validation cannot turn a negative scheduler/worker
    outcome into a promotable result.  Keeping this receipt as a separate
    input lets the reporting layer remain raw-trajectory blind while enforcing
    the protocol's hard proof boundary.
    """

    document, file_sha256 = _read_json_artifact(
        path, label=f"{slot} completion receipt"
    )
    source = _mapping(score.get("source"), label=f"{slot} score source")
    method = _mapping(score.get("method"), label=f"{slot} score method")
    expected_arm = _STRUCTURAL_POLICY[slot]
    schema = document.get("schema")
    try:
        if schema in {
            SIAM_NB4_CHTC_RECEIPT_SCHEMA_V3,
            SIAM_NB4_CHTC_RECEIPT_SCHEMA_V4,
        }:
            verified = validate_siam_nb4_completion_receipt(
                document,
                expected_source_revision=str(method.get("code_revision") or ""),
                expected_seed_sha256=str(source.get("seed_sha256") or ""),
                expected_trajectory_sha256=str(source.get("run_sha256") or ""),
                expected_arm=expected_arm,
            )
            segment_id = str(verified["segment"])
            if segment_id == SIAM_NB4_SCREEN_SEGMENT.segment_id:
                segment = SIAM_NB4_SCREEN_SEGMENT
                proof_role = "diagnostic_screen"
            elif segment_id == SIAM_NB4_CONTINUATION_SEGMENT.segment_id:
                segment = SIAM_NB4_CONTINUATION_SEGMENT
                proof_role = "diagnostic_continuation"
            else:  # The public receipt validator should already fail first.
                raise ValueError(f"unregistered raw segment {segment_id!r}")
            source_tree_files_sha256 = _sha256(
                verified.get("source_tree_files_sha256"),
                label=f"{slot} completion source-tree digest",
            )
            artifact = {
                "path": str(path.resolve()),
                "file_sha256": file_sha256,
                "schema": str(schema),
                "proof_kind": "raw_action",
                "package_id": str(verified["package_id"]),
                "action_id": str(verified["action_id"]),
                "trajectory_sha256": str(verified["trajectory_sha256"]),
                "proof_role": proof_role,
                "segment": segment_id,
                "initial_time": float(segment.times[0]),
                "terminal_time": float(segment.times[-1]),
                "point_count": len(segment.times),
                "manuscript_eligible": False,
                "source_tree_files_sha256": source_tree_files_sha256,
            }
        elif schema == SIAM_NB4_STITCH_COMPLETION_V1:
            verified = validate_siam_nb4_stitch_completion_receipt(
                document,
                expected_trajectory_sha256=str(source.get("run_sha256") or ""),
                expected_seed_sha256=str(source.get("seed_sha256") or ""),
                expected_arm=expected_arm,
            )
            expected_revision = str(method.get("code_revision") or "")
            source_segments = list(verified.get("source_segments") or ())
            if not source_segments or any(
                not isinstance(row, Mapping)
                or row.get("source_revision") != expected_revision
                for row in source_segments
            ):
                raise ValueError(
                    "stitched source revisions do not bind the scored method"
                )
            source_tree_digests = {
                _sha256(
                    _mapping(
                        row.get("completion_receipt"),
                        label=f"{slot} stitched source completion receipt",
                    ).get("source_tree_files_sha256"),
                    label=f"{slot} stitched source-tree digest",
                )
                for row in source_segments
            }
            if len(source_tree_digests) != 1:
                raise ValueError(
                    "stitched source receipts do not bind one source tree"
                )
            source_tree_files_sha256 = next(iter(source_tree_digests))
            artifact = {
                "path": str(path.resolve()),
                "file_sha256": file_sha256,
                "schema": SIAM_NB4_STITCH_COMPLETION_V1,
                "proof_kind": "stitched_trajectory",
                "completion_sha256": str(verified["completion_sha256"]),
                "trajectory_sha256": str(
                    verified["stitched_trajectory"]["sha256"]
                ),
                "proof_role": "full_horizon_stitched",
                "segment": _STITCHED_NB4_SEGMENT,
                "initial_time": 0.0,
                "terminal_time": SIAM_NB4_T_FINAL,
                "point_count": SIAM_NB4_POINT_COUNT,
                "manuscript_eligible": True,
                "source_tree_files_sha256": source_tree_files_sha256,
            }
        else:
            raise ValueError(
                "unsupported proof schema; require positive raw v3 or "
                "stitched-completion receipt"
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise SIAMResultsContractError(
            f"{slot} completion proof fails canonical validation: {exc}"
        ) from exc
    return document, artifact


def _pending_slot(slot: str) -> dict[str, Any]:
    return {
        "method": slot,
        "reader_label": _READER_LABEL[slot],
        "state": "pending",
        "qualification_basis": None,
        "individual_gate_pass": None,
        "individual_failed_gates": None,
        "pending_reason": "no_verified_completed_trajectory",
        "response_normalized_primitive_errors": None,
        "response_normalized_derived_screening_errors": None,
        "mean_infidelity": None,
        "energy_guard": None,
        "accepted_integration_steps": None,
        "terminal_n_theta": None,
        "terminal_graph_proxy": None,
        "completion_proof_role": None,
        "completion_segment": None,
        "terminal_time": None,
        "manuscript_eligible": None,
        "inputs": None,
    }


def _completed_slot(
    slot: str,
    *,
    score: Mapping[str, Any],
    score_artifact: Mapping[str, Any],
    resource: Mapping[str, Any],
    resource_artifact: Mapping[str, Any],
    completion_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _mapping(score.get("summary"), label=f"{slot} score summary")
    energy = _mapping(
        _mapping(score.get("diagnostics"), label=f"{slot} diagnostics").get(
            "energy"
        ),
        label=f"{slot} energy guard",
    )
    thresholds = _mapping(
        score.get("thresholds"), label=f"{slot} scoring thresholds"
    )
    individual_target = _finite(
        thresholds.get("individual_response_target"),
        label=f"{slot} individual-response target",
        nonnegative=True,
    )
    broad_guard = _finite(
        thresholds.get("broad_response_guard"),
        label=f"{slot} broad-response guard",
        nonnegative=True,
    )
    fidelity_ceiling = _finite(
        thresholds.get("mean_infidelity_ceiling"),
        label=f"{slot} fidelity ceiling",
        nonnegative=True,
    )
    energy_ceiling = _finite(
        thresholds.get("relative_energy_drift_ceiling"),
        label=f"{slot} energy-guard ceiling",
        nonnegative=True,
    )
    if min(individual_target, broad_guard, fidelity_ceiling, energy_ceiling) <= 0.0:
        raise SIAMResultsContractError(f"{slot} scoring ceilings must be positive.")
    response_floor = _finite(
        thresholds.get("response_floor"),
        label=f"{slot} response floor",
        nonnegative=True,
    )
    primitive_metrics = _verified_response_metric_set(
        score.get("primitive_response_metrics"),
        expected_names=_NB4_PRIMITIVE_NAMES,
        label=f"{slot} Nb4 primitive response metrics",
        response_floor=response_floor,
    )
    primitive_errors = {
        name: primitive_metrics[name][
            "response_normalized_mean_absolute_error"
        ]
        for name in _NB4_PRIMITIVE_NAMES
    }
    derived_metrics = _verified_response_metric_set(
        score.get("derived_screening_metrics"),
        expected_names=_NB4_DERIVED_NAMES,
        label=f"{slot} Nb4 derived screening metrics",
        response_floor=response_floor,
    )
    derived_errors = {
        name: derived_metrics[name]["response_normalized_mean_absolute_error"]
        for name in _NB4_DERIVED_NAMES
    }
    mean_infidelity = _finite(
        summary.get("mean_infidelity"),
        label=f"{slot} mean infidelity",
        nonnegative=True,
    )
    g_e = _finite(energy.get("g_E"), label=f"{slot} g_E", nonnegative=True)
    expected_individual = all(
        value <= individual_target for value in primitive_errors.values()
    )
    expected_broad = all(value <= broad_guard for value in primitive_errors.values())
    expected_fidelity = mean_infidelity <= fidelity_ceiling
    expected_energy = g_e <= energy_ceiling
    declared_gates = {
        "individual_target_pass": _boolean(
            summary.get("individual_target_pass"),
            label=f"{slot} individual-target gate",
        ),
        "broad_response_guard_pass": _boolean(
            summary.get("broad_response_guard_pass"),
            label=f"{slot} broad-response gate",
        ),
        "fidelity_pass": _boolean(
            summary.get("fidelity_pass"), label=f"{slot} fidelity gate"
        ),
        "energy_conservation_pass": _boolean(
            summary.get("energy_conservation_pass"),
            label=f"{slot} energy-conservation gate",
        ),
    }
    expected_gates = {
        # The scorer's two response-pass fields are admission decisions, so
        # they already include the fidelity and energy safeguards.  Preserve
        # that contract here while reporting the three ingredients separately
        # below.
        "individual_target_pass": (
            expected_individual and expected_fidelity and expected_energy
        ),
        "broad_response_guard_pass": (
            expected_broad and expected_fidelity and expected_energy
        ),
        "fidelity_pass": expected_fidelity,
        "energy_conservation_pass": expected_energy,
    }
    if declared_gates != expected_gates or _boolean(
        energy.get("pass"), label=f"{slot} energy diagnostic pass"
    ) != expected_energy:
        raise SIAMResultsContractError(
            f"{slot} score gate declarations are inconsistent with its metrics."
        )
    gates = {
        "individual_response_target_pass": expected_individual,
        "fidelity_ceiling_pass": expected_fidelity,
        "energy_guard_pass": expected_energy,
    }
    gates["all_declared_gates_pass"] = bool(all(gates.values()))
    individual_gate_pass = bool(gates["all_declared_gates_pass"])
    state = "qualified" if individual_gate_pass else "candidate"
    individual_failed_gates = sorted(
        name
        for name, passed in gates.items()
        if name != "all_declared_gates_pass" and not passed
    )

    tuple_order = list(resource.get("canonical_tuple_order") or ())
    expected_order = ["C_hat_2q", "C_hat_d", "C_hat_1q", "N_theta"]
    if tuple_order != expected_order:
        raise SIAMResultsContractError(
            f"{slot} terminal-resource tuple order is not canonical."
        )
    terminal_tuple = _mapping(
        resource.get("terminal_resource_tuple"),
        label=f"{slot} terminal resource tuple",
    )
    graph = [
        _finite(
            terminal_tuple.get(name),
            label=f"{slot} terminal {name}",
            nonnegative=True,
        )
        for name in ("C_hat_2q", "C_hat_d", "C_hat_1q")
    ]
    n_theta = _integer(
        terminal_tuple.get("N_theta"),
        label=f"{slot} terminal N_theta",
        nonnegative=True,
    )
    if n_theta != _integer(
        resource.get("terminal_logical_macro_count"),
        label=f"{slot} terminal logical macro count",
        nonnegative=True,
    ):
        raise SIAMResultsContractError(
            f"{slot} terminal N_theta differs from its logical macro count."
        )
    accepted_steps = _integer(
        resource.get("accepted_integration_steps"),
        label=f"{slot} accepted integration steps",
        nonnegative=True,
    )
    terminal_time = _finite(
        resource.get("terminal_time"), label=f"{slot} terminal time"
    )
    grid = _mapping(score.get("grid"), label=f"{slot} score grid")
    times = list(grid.get("times") or ())
    if not times or not math.isclose(
        terminal_time,
        _finite(times[-1], label=f"{slot} final score-grid time"),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise SIAMResultsContractError(
            f"{slot} terminal resource and score end on different times."
        )
    proof_initial_time = _finite(
        completion_artifact.get("initial_time"),
        label=f"{slot} completion-proof initial time",
    )
    proof_terminal_time = _finite(
        completion_artifact.get("terminal_time"),
        label=f"{slot} completion-proof terminal time",
    )
    proof_point_count = _integer(
        completion_artifact.get("point_count"),
        label=f"{slot} completion-proof point count",
        nonnegative=True,
    )
    if (
        proof_point_count != len(times)
        or not math.isclose(
            proof_initial_time,
            _finite(times[0], label=f"{slot} initial score-grid time"),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            proof_terminal_time,
            terminal_time,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise SIAMResultsContractError(
            f"{slot} completion proof and scored grid do not identify one segment."
        )

    return {
        "method": slot,
        "reader_label": _READER_LABEL[slot],
        "state": state,
        "qualification_basis": (
            "individual_target" if individual_gate_pass else None
        ),
        "individual_gate_pass": individual_gate_pass,
        "individual_failed_gates": individual_failed_gates,
        "pending_reason": None,
        # Compatibility alias for the first results-contract draft.  These are
        # always individual-gate misses even when a matched pair later provides
        # the slot's qualification basis.
        "failed_gates": individual_failed_gates,
        "response_normalized_primitive_errors": primitive_errors,
        "response_normalized_derived_screening_errors": derived_errors,
        "mean_infidelity": mean_infidelity,
        "energy_guard": {
            "g_E": g_e,
            "ceiling": energy_ceiling,
            "pass": expected_energy,
        },
        "gates": gates,
        "accepted_integration_steps": accepted_steps,
        "terminal_n_theta": n_theta,
        "terminal_graph_proxy": graph,
        "terminal_graph_proxy_order": ["C_hat_2q", "C_hat_d", "C_hat_1q"],
        "completion_proof_role": str(completion_artifact["proof_role"]),
        "completion_segment": str(completion_artifact["segment"]),
        "terminal_time": terminal_time,
        "manuscript_eligible": _boolean(
            completion_artifact.get("manuscript_eligible"),
            label=f"{slot} manuscript eligibility",
        ),
        "inputs": {
            "score": dict(score_artifact),
            "terminal_resource": dict(resource_artifact),
            "completion_receipt": dict(completion_artifact),
            "trajectory_sha256": str(score["source"]["run_sha256"]),
            "seed_sha256": str(score["source"]["seed_sha256"]),
        },
    }


def _comparison_identity(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = scores[0]
    reference = _mapping(first.get("reference"), label="SIAM exact reference")
    grid = _mapping(first.get("grid"), label="SIAM score grid")
    source = _mapping(first.get("source"), label="SIAM score source")
    method = _mapping(first.get("method"), label="SIAM score method")
    times = list(grid.get("times") or ())
    if not times:
        raise SIAMResultsContractError("SIAM promotion requires a nonempty grid.")
    identity = {
        "seed_sha256": str(source["seed_sha256"]),
        "post_quench_hamiltonian_sha256": str(
            source["post_quench_hamiltonian_sha256"]
        ),
        "comparison_fingerprint": str(method["comparison_fingerprint"]),
        "physics_fingerprint": str(method["physics_fingerprint"]),
        "exact_reference": {
            key: str(reference[key])
            for key in (
                "key",
                "manifest_sha256",
                "data_sha256",
                "store_schema",
                "reference_schema",
                "observable_schema",
                "propagator_schema",
            )
        },
        "grid": {
            "ordered_values_sha256": str(grid["ordered_values_sha256"]),
            "point_count": _integer(
                grid["point_count"], label="SIAM score-grid point count"
            ),
            "initial_time": _finite(times[0], label="SIAM initial grid time"),
            "terminal_time": _finite(times[-1], label="SIAM terminal grid time"),
        },
    }
    for other in scores[1:]:
        other_identity = _comparison_identity([other])
        if identity != other_identity:
            raise SIAMResultsContractError(
                "Completed SIAM arms do not share one exact comparison identity."
            )
    return identity


def _revision_only_pair_incomparability(
    scores: Mapping[str, Mapping[str, Any]],
    completion_artifacts: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Return the shared source-tree digest for a revision-only mismatch.

    This is deliberately narrower than matched-pair admission.  It exists only
    so reporting can retain two verified numerical rows while the canonical
    classifier continues to reject the pair's cost comparison.
    """

    if set(scores) != set(_METHODS) or set(completion_artifacts) != set(_METHODS):
        return None
    avqds = scores["avqds"]
    full_apm = scores["full_apm"]
    avqds_method = _mapping(avqds.get("method"), label="AVQDS score method")
    apm_method = _mapping(full_apm.get("method"), label="full-APM score method")
    if avqds_method.get("code_revision") == apm_method.get("code_revision"):
        return None

    comparable_fields = (
        ("method", "comparison_fingerprint"),
        ("source", "seed_sha256"),
        ("reference", "key"),
        ("reference", "manifest_sha256"),
        ("reference", "data_sha256"),
        ("grid", "ordered_values_sha256"),
        ("grid", "point_count"),
    )
    for section, field in comparable_fields:
        left = _mapping(avqds.get(section), label=f"AVQDS {section}")
        right = _mapping(full_apm.get(section), label=f"full-APM {section}")
        if left.get(field) != right.get(field):
            return None
    if avqds.get("thresholds") != full_apm.get("thresholds") or avqds.get(
        "accuracy_policy"
    ) != full_apm.get("accuracy_policy"):
        return None

    digests = {
        _sha256(
            completion_artifacts[slot].get("source_tree_files_sha256"),
            label=f"{slot} completion source-tree digest",
        )
        for slot in _METHODS
    }
    return next(iter(digests)) if len(digests) == 1 else None


def _format_scientific(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.3e}"


def _format_graph(values: Sequence[float]) -> str:
    return "(" + ",".join(f"{value:.0f}" for value in values) + ")"


def _canonical_promotion_role(value: str) -> str:
    role = str(value).strip().lower()
    if role not in {
        SIAM_RESULTS_DIAGNOSTIC_ROLE,
        SIAM_RESULTS_MANUSCRIPT_ROLE,
    }:
        raise SIAMResultsContractError(
            "SIAM promotion role must be diagnostic_screening or manuscript_table."
        )
    return role


def render_siam_results_latex_fragment(contract: Mapping[str, Any]) -> str:
    """Render a deterministic compact table without consulting other files."""

    slots = _mapping(contract.get("slots"), label="SIAM promotion slots")
    completed = [
        _mapping(slots[name], label=f"SIAM {name} slot")
        for name in _METHODS
        if _mapping(slots[name], label=f"SIAM {name} slot").get("state")
        != "pending"
    ]
    primitive_names = (
        list(completed[0]["response_normalized_primitive_errors"])
        if completed
        else []
    )
    primitive_header = ",".join(name.replace("_", r"\_") for name in primitive_names)
    lines = [
        "% Generated by siam_results_contract.py; do not edit by hand.",
        f"% schema: {SIAM_RESULTS_LATEX_FRAGMENT_V2}",
        r"\begin{tabular}{llcccccc}",
        r"\hline",
        (
            r"Method & State & $\epsilon_{\rm resp}$"
            + (f" ({primitive_header})" if primitive_header else "")
            + r" & $\overline{1-F}$ & $g_E$ & $N_{\rm step}$"
            + r" & $N_\theta$ & $(\widehat C_{2q},\widehat C_D,\widehat C_{1q})$ \\"
        ),
        r"\hline",
        r"Exact & reference & 0 & 0 & 0 & -- & -- & -- \\",
    ]
    for name in _METHODS:
        slot = _mapping(slots[name], label=f"SIAM {name} slot")
        label = str(slot["reader_label"])
        state = str(slot["state"])
        if state == "pending":
            lines.append(
                rf"{label} & pending & -- & -- & -- & -- & -- & -- \\"
            )
            continue
        errors = _mapping(
            slot["response_normalized_primitive_errors"],
            label=f"SIAM {name} response errors",
        )
        if list(errors) != primitive_names:
            raise SIAMResultsContractError(
                "Completed SIAM arms expose different primitive-error schemas."
            )
        response_tuple = "(" + ",".join(
            _format_scientific(float(errors[key])) for key in primitive_names
        ) + ")"
        lines.append(
            " & ".join(
                (
                    label,
                    state,
                    f"${response_tuple}$",
                    _format_scientific(float(slot["mean_infidelity"])),
                    _format_scientific(float(slot["energy_guard"]["g_E"])),
                    f"{int(slot['accepted_integration_steps']):,}",
                    str(int(slot["terminal_n_theta"])),
                    f"${_format_graph(slot['terminal_graph_proxy'])}$ " + r"\\",
                )
            )
        )
    pair = _mapping(
        contract.get("pair_comparison"), label="SIAM pair comparison"
    )
    verdict = str(pair.get("reader_verdict") or "")
    allowed_verdicts = {
        "individually qualified + matched",
        "matched",
        "accuracy-confounded",
        "pending",
        "pending - incomparable revisions",
    }
    if verdict not in allowed_verdicts:
        raise SIAMResultsContractError("SIAM pair comparison lacks a reader verdict.")
    lines.append(
        rf"\multicolumn{{8}}{{l}}{{Pair verdict: {verdict}.}} \\"
    )
    required_resolution = pair.get("required_resolution")
    if required_resolution is not None:
        resolution = str(required_resolution).strip()
        if not resolution:
            raise SIAMResultsContractError(
                "SIAM pair comparison has an empty required resolution."
            )
        lines.append(rf"\multicolumn{{8}}{{l}}{{{resolution}}} \\")
    role = _canonical_promotion_role(str(contract.get("promotion_role") or ""))
    manuscript = _mapping(
        contract.get("manuscript_contract"), label="SIAM manuscript contract"
    )
    if role == SIAM_RESULTS_MANUSCRIPT_ROLE:
        if manuscript.get("validated_stitched_t5_required") is not True or (
            manuscript.get("all_numerical_slots_validated_stitched_t5") is not True
        ):
            raise SIAMResultsContractError(
                "SIAM manuscript table lacks validated stitched-T=5 evidence."
            )
        evidence_label = "validated stitched $T=5$ manuscript evidence"
    else:
        evidence_label = "diagnostic screening evidence; not manuscript promotion"
    lines.append(rf"\multicolumn{{8}}{{l}}{{Evidence role: {evidence_label}.}} \\")
    lines.extend((r"\hline", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def render_siam_results_pdf(contract: Mapping[str, Any]) -> bytes:
    """Render the reader-facing SIAM results table as a deterministic PDF.

    This is the canonical middle-ground artifact: it mirrors both method slots,
    retains numerical candidates, keeps unfinished arms as pending, and exposes
    every primitive response-normalized error before any manuscript promotion.
    The renderer consumes only the verified promotion contract and never opens
    raw trajectories.
    """

    payload = dict(contract)
    digest = payload.pop("promotion_sha256", None)
    if payload.get("schema") != SIAM_RESULTS_PROMOTION_V2 or digest != (
        _document_sha256(payload)
    ):
        raise SIAMResultsContractError("SIAM promotion-receipt digest mismatch.")
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Rendering the SIAM results PDF requires reportlab."
        ) from exc

    slots = _mapping(payload.get("slots"), label="SIAM promotion slots")
    pair = _mapping(
        payload.get("pair_comparison"), label="SIAM pair comparison"
    )
    identity = _mapping(
        payload.get("comparison_identity"), label="SIAM comparison identity"
    )
    grid = _mapping(identity.get("grid"), label="SIAM comparison grid")
    buffer = io.BytesIO()
    page_width, page_height = landscape(letter)
    pdf = canvas.Canvas(
        buffer,
        pagesize=(page_width, page_height),
        pageCompression=1,
        invariant=1,
    )
    pdf.setTitle("SIAM screening benchmark results")
    pdf.setAuthor("Holstein time-dynamics benchmark pipeline")

    margin = 34.0
    pdf.setFillColor(colors.HexColor("#17365D"))
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(margin, page_height - 38.0, "SIAM screening benchmark results")
    pdf.setFillColor(colors.HexColor("#404040"))
    pdf.setFont("Helvetica", 8.5)
    role = _canonical_promotion_role(str(payload.get("promotion_role") or ""))
    role_label = (
        "validated stitched T=5 manuscript evidence"
        if role == SIAM_RESULTS_MANUSCRIPT_ROLE
        else "diagnostic screening evidence - not manuscript promotion"
    )
    pdf.drawString(
        margin,
        page_height - 54.0,
        (
            f"Role: {role_label} | grid: {int(grid['point_count'])} points, "
            f"t={float(grid['initial_time']):g}..{float(grid['terminal_time']):g}"
        ),
    )
    pdf.drawRightString(
        page_width - margin,
        page_height - 54.0,
        f"cell {str(payload.get('cell_id') or '')[:12]}",
    )

    headers = (
        "Method",
        "State",
        "m_d",
        "D_d",
        "C_1",
        "C_2",
        "C_3",
        "C_4",
        "mean 1-F",
        "g_E",
        "steps",
        "N_theta",
        "(C2q,CD,C1q)",
    )
    widths = (60, 55, 45, 45, 45, 45, 45, 45, 65, 52, 55, 45, 108)
    table_width = float(sum(widths))
    x0 = (page_width - table_width) / 2.0
    y_top = page_height - 88.0
    header_height = 30.0
    row_height = 34.0

    pdf.setFillColor(colors.HexColor("#17365D"))
    pdf.rect(x0, y_top - header_height, table_width, header_height, stroke=0, fill=1)
    x = x0
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 7.2)
    for header, width in zip(headers, widths, strict=True):
        pdf.drawCentredString(x + width / 2.0, y_top - 18.5, header)
        x += width

    rows: list[tuple[str, Mapping[str, Any] | None]] = [
        ("Exact", None),
        ("AVQDS", _mapping(slots["avqds"], label="SIAM AVQDS slot")),
        ("Full APM", _mapping(slots["full_apm"], label="SIAM full-APM slot")),
    ]
    state_colors = {
        "reference": colors.HexColor("#D9EAF7"),
        "qualified": colors.HexColor("#DDEEDB"),
        "candidate": colors.HexColor("#FFF0CC"),
        "pending": colors.HexColor("#ECECEC"),
    }
    for row_index, (label, slot) in enumerate(rows):
        y = y_top - header_height - (row_index + 1) * row_height
        state = "reference" if slot is None else str(slot["state"])
        pdf.setFillColor(state_colors[state])
        pdf.rect(x0, y, table_width, row_height, stroke=0, fill=1)
        if slot is None:
            values = (label, state, "0", "0", "0", "0", "0", "0", "0", "0", "--", "--", "--")
        elif state == "pending":
            values = (label, state, "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--")
        else:
            errors = _mapping(
                slot.get("response_normalized_primitive_errors"),
                label=f"SIAM {label} primitive errors",
            )
            values = (
                label,
                state,
                *(_format_scientific(float(errors[name])) for name in _NB4_PRIMITIVE_NAMES),
                _format_scientific(float(slot["mean_infidelity"])),
                _format_scientific(float(slot["energy_guard"]["g_E"])),
                f"{int(slot['accepted_integration_steps']):,}",
                str(int(slot["terminal_n_theta"])),
                _format_graph(slot["terminal_graph_proxy"]),
            )
        x = x0
        pdf.setFillColor(colors.HexColor("#202020"))
        for column, (value, width) in enumerate(zip(values, widths, strict=True)):
            pdf.setFont(
                "Helvetica-Bold" if column in {0, 1} else "Helvetica",
                7.4 if column != 12 else 6.8,
            )
            pdf.drawCentredString(x + width / 2.0, y + 12.0, str(value))
            x += width

    table_bottom = y_top - header_height - len(rows) * row_height
    pdf.setStrokeColor(colors.HexColor("#8A8A8A"))
    pdf.setLineWidth(0.45)
    x = x0
    for width in widths:
        pdf.line(x, table_bottom, x, y_top)
        x += width
    pdf.line(x, table_bottom, x, y_top)
    pdf.line(x0, y_top, x0 + table_width, y_top)
    pdf.line(
        x0,
        y_top - header_height,
        x0 + table_width,
        y_top - header_height,
    )
    for row_index in range(1, len(rows) + 1):
        y = y_top - header_height - row_index * row_height
        pdf.line(x0, y, x0 + table_width, y)

    note_y = table_bottom - 24.0
    pdf.setFillColor(colors.HexColor("#202020"))
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(
        margin,
        note_y,
        f"Pair verdict: {str(pair.get('reader_verdict') or 'pending')}.",
    )
    pdf.setFont("Helvetica", 8)
    comparison_grid = _mapping(
        _mapping(
            payload.get("comparison_identity"),
            label="SIAM PDF comparison identity",
        ).get("grid"),
        label="SIAM PDF declared grid",
    )
    grid_initial = _finite(
        comparison_grid.get("initial_time"), label="SIAM PDF initial grid time"
    )
    grid_terminal = _finite(
        comparison_grid.get("terminal_time"), label="SIAM PDF terminal grid time"
    )
    pdf.drawString(
        margin,
        note_y - 16.0,
        "Errors are declared-grid "
        f"(t={grid_initial:g} to {grid_terminal:g}) response-normalized MAEs; "
        "candidates retain numerical values.",
    )
    pdf.drawString(
        margin,
        note_y - 29.0,
        "Canonical resources are logical N_theta and the deterministic graph proxy; no Qiskit compilation is used.",
    )
    pdf.setFillColor(colors.HexColor("#666666"))
    pdf.setFont("Helvetica", 7)
    pdf.drawRightString(
        page_width - margin,
        18.0,
        f"promotion {str(digest)[:12]} | page 1/1",
    )
    pdf.showPage()
    pdf.save()
    content = buffer.getvalue()
    if not content.startswith(b"%PDF-"):
        raise SIAMResultsContractError("SIAM results renderer did not produce PDF bytes.")
    return content


def build_siam_results_pdf_receipt(
    promotion: Mapping[str, Any], pdf_bytes: bytes
) -> dict[str, Any]:
    """Bind one deterministic PDF to the verified results promotion receipt."""

    payload = dict(promotion)
    promotion_sha256 = payload.pop("promotion_sha256", None)
    if payload.get("schema") != SIAM_RESULTS_PROMOTION_V2 or promotion_sha256 != (
        _document_sha256(payload)
    ):
        raise SIAMResultsContractError("SIAM promotion-receipt digest mismatch.")
    if not isinstance(pdf_bytes, bytes) or not pdf_bytes.startswith(b"%PDF-"):
        raise SIAMResultsContractError("SIAM results PDF bytes are invalid.")
    core = {
        "schema": SIAM_RESULTS_PDF_RECEIPT_V2,
        "render_schema": SIAM_RESULTS_PDF_V2,
        "promotion_sha256": str(promotion_sha256),
        "cell_id": str(payload["cell_id"]),
        "promotion_role": str(payload["promotion_role"]),
        "pdf": {
            "bytes": len(pdf_bytes),
            "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
            "page_count": 1,
        },
        "source_contract": "verified_results_promotion_only_no_raw_trajectory",
        "qiskit_compilation_used": False,
    }
    return {**core, "receipt_sha256": _document_sha256(core)}


def write_siam_results_pdf_artifact(
    *,
    pdf_path: str | Path,
    receipt_path: str | Path,
    pdf_bytes: bytes,
    receipt: Mapping[str, Any],
) -> None:
    """Install one PDF and digest receipt atomically and idempotently."""

    document = dict(receipt)
    digest = document.pop("receipt_sha256", None)
    pdf_record = _mapping(document.get("pdf"), label="SIAM results PDF record")
    if (
        document.get("schema") != SIAM_RESULTS_PDF_RECEIPT_V2
        or document.get("render_schema") != SIAM_RESULTS_PDF_V2
        or digest != _document_sha256(document)
        or pdf_record.get("bytes") != len(pdf_bytes)
        or pdf_record.get("sha256") != hashlib.sha256(pdf_bytes).hexdigest()
        or pdf_record.get("page_count") != 1
    ):
        raise SIAMResultsContractError("SIAM results-PDF receipt mismatch.")
    receipt_bytes = (
        json.dumps(dict(receipt), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    destinations = (
        (Path(pdf_path), pdf_bytes),
        (Path(receipt_path), receipt_bytes),
    )
    for path, content in destinations:
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(
                f"Refusing to replace immutable SIAM result: {path}"
            )
    for path, content in destinations:
        _write_immutable(path, content)


def build_siam_results_promotion(
    *,
    avqds_score_path: str | Path | None = None,
    avqds_resource_path: str | Path | None = None,
    avqds_completion_receipt_path: str | Path | None = None,
    full_apm_score_path: str | Path | None = None,
    full_apm_resource_path: str | Path | None = None,
    full_apm_completion_receipt_path: str | Path | None = None,
    promotion_role: str = SIAM_RESULTS_DIAGNOSTIC_ROLE,
) -> tuple[dict[str, Any], str]:
    """Build one deterministic promotion receipt and LaTeX table fragment."""

    role = _canonical_promotion_role(promotion_role)

    paths = {
        "avqds": (
            avqds_score_path,
            avqds_resource_path,
            avqds_completion_receipt_path,
        ),
        "full_apm": (
            full_apm_score_path,
            full_apm_resource_path,
            full_apm_completion_receipt_path,
        ),
    }
    slots: dict[str, dict[str, Any]] = {}
    verified_documents: dict[str, dict[str, Any]] = {}
    original_score_documents: dict[str, dict[str, Any]] = {}
    completion_artifacts: dict[str, dict[str, Any]] = {}
    for slot, (score_value, resource_value, completion_value) in paths.items():
        presence = tuple(
            value is not None
            for value in (score_value, resource_value, completion_value)
        )
        if any(presence) and not all(presence):
            raise SIAMResultsContractError(
                f"{slot} requires score, terminal-resource, and positive "
                "completion-receipt sidecars."
            )
        if score_value is None:
            slots[slot] = _pending_slot(slot)
            continue
        score_path = Path(score_value)
        resource_path = Path(resource_value)
        score, score_artifact, score_document = _verified_score_artifact(
            score_path, expected_slot=slot
        )
        original_score_documents[slot] = score_document
        resource, resource_artifact = _verified_resource_artifact(
            resource_path, slot=slot, score=score
        )
        assert completion_value is not None
        _completion, completion_artifact = _verified_completion_artifact(
            Path(completion_value), slot=slot, score=score
        )
        completion_artifacts[slot] = completion_artifact
        verified_documents[slot] = score
        slots[slot] = _completed_slot(
            slot,
            score=score,
            score_artifact=score_artifact,
            resource=resource,
            resource_artifact=resource_artifact,
            completion_artifact=completion_artifact,
        )
    if not verified_documents:
        raise SIAMResultsContractError(
            "At least one verified completed SIAM arm is required to bind "
            "the exact cell."
        )

    completed_slots = [
        slot for slot in slots.values() if slot["state"] != "pending"
    ]
    all_numerical_slots_validated_stitched_t5 = bool(
        completed_slots
        and all(
            slot["manuscript_eligible"] is True
            and slot["completion_proof_role"] == "full_horizon_stitched"
            and slot["completion_segment"] == _STITCHED_NB4_SEGMENT
            and math.isclose(
                float(slot["terminal_time"]),
                SIAM_NB4_T_FINAL,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for slot in completed_slots
        )
    )
    if (
        role == SIAM_RESULTS_MANUSCRIPT_ROLE
        and not all_numerical_slots_validated_stitched_t5
    ):
        raise SIAMResultsContractError(
            "SIAM manuscript promotion requires a validated stitched T=5 "
            "completion proof for every numerical slot."
        )

    identity = _comparison_identity(list(verified_documents.values()))
    primitive_names = list(
        next(
            slot["response_normalized_primitive_errors"]
            for slot in slots.values()
            if slot["state"] != "pending"
        )
    )
    exact_slot = {
        "state": "reference",
        **identity["exact_reference"],
        "response_normalized_primitive_errors": {
            name: 0.0 for name in primitive_names
        },
        "response_normalized_derived_screening_errors": {
            name: 0.0 for name in _NB4_DERIVED_NAMES
        },
        "mean_infidelity": 0.0,
        "energy_guard_g_E": 0.0,
    }
    if len(verified_documents) == 2:
        completion_source_tree_digests = {
            _sha256(
                completion_artifacts[slot].get("source_tree_files_sha256"),
                label=f"{slot} completion source-tree digest",
            )
            for slot in _METHODS
        }
        if len(completion_source_tree_digests) != 1:
            raise SIAMResultsContractError(
                "Completed SIAM arms bind different source-tree digests."
            )
        try:
            pair = classify_siam_matched_pair(
                original_score_documents["full_apm"],
                original_score_documents["avqds"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            shared_source_tree_sha256 = _revision_only_pair_incomparability(
                verified_documents,
                completion_artifacts,
            )
            if shared_source_tree_sha256 is None:
                raise SIAMResultsContractError(
                    "Completed SIAM arms fail matched-pair verification."
                ) from exc
            for slot in _METHODS:
                completed_slot = slots[slot]
                completed_slot["state"] = "candidate"
                completed_slot["qualification_basis"] = None
            pair_comparison = {
                "state": "pending_incomparable",
                "classification": "pending_incomparable_code_revision",
                "cost_comparison_admissible": False,
                "pending_reason": "method_code_revision_mismatch",
                "reader_verdict": "pending - incomparable revisions",
                "shared_source_tree_files_sha256": shared_source_tree_sha256,
                "required_resolution": (
                    "Final stitched continuations must share one code revision "
                    "before matched-cost claims."
                ),
            }
        else:
            pair_comparison = {
                "state": "complete",
                "pair_sha256": str(pair["pair_sha256"]),
                **dict(pair["matched_accuracy"]),
            }
            if pair_comparison["cost_comparison_admissible"]:
                for slot in _METHODS:
                    completed_slot = slots[slot]
                    completed_slot["state"] = "qualified"
                    completed_slot["qualification_basis"] = (
                        "individual_target"
                        if completed_slot["individual_gate_pass"]
                        else "matched_pair"
                    )
            else:
                # Once both arms exist, the comparison—not either isolated
                # lane—is the reader-facing unit.  An unmatched pair is
                # accuracy-confounded, so both numerical rows remain visible
                # but neither may retain a publication-facing qualified label.
                for slot in _METHODS:
                    completed_slot = slots[slot]
                    completed_slot["state"] = "candidate"
                    completed_slot["qualification_basis"] = None
            pair_comparison["reader_verdict"] = (
                "individually qualified + matched"
                if pair_comparison["cost_comparison_admissible"]
                and all(slots[slot]["individual_gate_pass"] for slot in _METHODS)
                else "matched"
                if pair_comparison["cost_comparison_admissible"]
                else "accuracy-confounded"
            )
    else:
        pair_comparison = {
            "state": "pending",
            "classification": "pending",
            "cost_comparison_admissible": False,
            "pending_reason": "one_method_slot_has_no_verified_completed_trajectory",
            "reader_verdict": "pending",
        }

    identity_core = {
        "seed_sha256": identity["seed_sha256"],
        "post_quench_hamiltonian_sha256": identity[
            "post_quench_hamiltonian_sha256"
        ],
        "comparison_fingerprint": identity["comparison_fingerprint"],
        "physics_fingerprint": identity["physics_fingerprint"],
        "exact_reference": identity["exact_reference"],
        "grid": identity["grid"],
    }
    cell_id = _document_sha256(identity_core)
    counts = {
        "qualified": sum(slot["state"] == "qualified" for slot in slots.values()),
        "candidate": sum(slot["state"] == "candidate" for slot in slots.values()),
        "pending": sum(slot["state"] == "pending" for slot in slots.values()),
    }
    core: dict[str, Any] = {
        "schema": SIAM_RESULTS_PROMOTION_V2,
        "observable_contract": dict(
            next(iter(verified_documents.values()))["observable_contract"]
        ),
        "promotion_role": role,
        "manuscript_contract": {
            "validated_stitched_t5_required": (
                role == SIAM_RESULTS_MANUSCRIPT_ROLE
            ),
            "all_numerical_slots_validated_stitched_t5": (
                all_numerical_slots_validated_stitched_t5
            ),
            "screen_t1_role": "diagnostic_only",
        },
        "cell_id": cell_id,
        "comparison_identity": identity_core,
        "accuracy_policy": dict(
            next(iter(verified_documents.values()))["accuracy_policy"]
        ),
        "exact": exact_slot,
        "slots": slots,
        "pair_comparison": pair_comparison,
        "counts": counts,
        "canonical_resource_accounting": {
            "terminal_n_theta": "independent_logical_shared_coordinates",
            "terminal_graph_proxy_order": [
                "C_hat_2q",
                "C_hat_d",
                "C_hat_1q",
            ],
            "qiskit_compilation_required": False,
            "qiskit_compilation_role": "optional_diagnostic_only",
        },
    }
    latex = render_siam_results_latex_fragment(core)
    core["latex_fragment"] = {
        "schema": SIAM_RESULTS_LATEX_FRAGMENT_V2,
        "sha256": hashlib.sha256(latex.encode("utf-8")).hexdigest(),
    }
    return {**core, "promotion_sha256": _document_sha256(core)}, latex


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"Refusing to replace immutable SIAM result: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(
                    f"Refusing to replace immutable SIAM result: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def write_siam_results_promotion(
    *,
    receipt_path: str | Path,
    latex_path: str | Path,
    receipt: Mapping[str, Any],
    latex: str,
) -> None:
    """Install a digest-bound receipt and its deterministic table idempotently."""

    payload = dict(receipt)
    digest = payload.pop("promotion_sha256", None)
    if payload.get("schema") != SIAM_RESULTS_PROMOTION_V2 or digest != (
        _document_sha256(payload)
    ):
        raise SIAMResultsContractError("SIAM promotion-receipt digest mismatch.")
    latex_record = _mapping(
        payload.get("latex_fragment"), label="SIAM LaTeX fragment receipt"
    )
    latex_bytes = latex.encode("utf-8")
    if latex_record.get("schema") != SIAM_RESULTS_LATEX_FRAGMENT_V2 or (
        latex_record.get("sha256") != hashlib.sha256(latex_bytes).hexdigest()
    ):
        raise SIAMResultsContractError("SIAM LaTeX fragment digest mismatch.")
    receipt_bytes = (
        json.dumps(dict(receipt), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    destinations = (
        (Path(latex_path), latex_bytes),
        (Path(receipt_path), receipt_bytes),
    )
    for path, content in destinations:
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(
                f"Refusing to replace immutable SIAM result: {path}"
            )
    for path, content in destinations:
        _write_immutable(path, content)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avqds-score", type=Path)
    parser.add_argument("--avqds-resource", type=Path)
    parser.add_argument("--avqds-completion-receipt", type=Path)
    parser.add_argument("--full-apm-score", type=Path)
    parser.add_argument("--full-apm-resource", type=Path)
    parser.add_argument("--full-apm-completion-receipt", type=Path)
    parser.add_argument(
        "--promotion-role",
        choices=(SIAM_RESULTS_DIAGNOSTIC_ROLE, SIAM_RESULTS_MANUSCRIPT_ROLE),
        default=SIAM_RESULTS_DIAGNOSTIC_ROLE,
        help=(
            "diagnostic_screening permits validated segment evidence; "
            "manuscript_table requires stitched T=5 completion proofs"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-pdf-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt, latex = build_siam_results_promotion(
        avqds_score_path=args.avqds_score,
        avqds_resource_path=args.avqds_resource,
        avqds_completion_receipt_path=args.avqds_completion_receipt,
        full_apm_score_path=args.full_apm_score,
        full_apm_resource_path=args.full_apm_resource,
        full_apm_completion_receipt_path=args.full_apm_completion_receipt,
        promotion_role=args.promotion_role,
    )
    write_siam_results_promotion(
        receipt_path=args.output_json,
        latex_path=args.output_tex,
        receipt=receipt,
        latex=latex,
    )
    pdf_bytes = render_siam_results_pdf(receipt)
    pdf_receipt = build_siam_results_pdf_receipt(receipt, pdf_bytes)
    write_siam_results_pdf_artifact(
        pdf_path=args.output_pdf,
        receipt_path=args.output_pdf_receipt,
        pdf_bytes=pdf_bytes,
        receipt=pdf_receipt,
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_tex": str(args.output_tex),
                "output_pdf": str(args.output_pdf),
                "output_pdf_receipt": str(args.output_pdf_receipt),
                "promotion_sha256": receipt["promotion_sha256"],
                "pdf_sha256": pdf_receipt["pdf"]["sha256"],
                "counts": receipt["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SIAM_RESULTS_LATEX_FRAGMENT_V1",
    "SIAM_RESULTS_LATEX_FRAGMENT_V2",
    "SIAM_RESULTS_DIAGNOSTIC_ROLE",
    "SIAM_RESULTS_MANUSCRIPT_ROLE",
    "SIAM_RESULTS_PDF_RECEIPT_V1",
    "SIAM_RESULTS_PDF_RECEIPT_V2",
    "SIAM_RESULTS_PDF_V1",
    "SIAM_RESULTS_PDF_V2",
    "SIAM_RESULTS_PROMOTION_V1",
    "SIAM_RESULTS_PROMOTION_V2",
    "SIAMResultsContractError",
    "build_siam_results_promotion",
    "build_siam_results_pdf_receipt",
    "render_siam_results_latex_fragment",
    "render_siam_results_pdf",
    "write_siam_results_pdf_artifact",
    "write_siam_results_promotion",
]
