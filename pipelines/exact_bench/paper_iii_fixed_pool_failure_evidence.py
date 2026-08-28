#!/usr/bin/env python3
"""Acquire native Paper-III fixed-pool selected-QSE termination evidence.

This driver runs only the canonical selected-QSE route.  It never invokes
Davidson, Krylov, or a fixed-class comparator.  Online terminal status is
derived solely from the selector receipt; exact root errors are stored under a
separate post-run audit namespace and cannot rewrite that status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.exact_bench.paper_iii_growth_trace_campaign import (
    DEFAULT_DIMENSION_CAP,
    RESIDUAL_MARKER,
    _Q0_POLICY,
    trace_selected_qse_fixed_pool,
)
from pipelines.exact_bench.paper_iii_matched_accuracy_campaign import REGIME_SETS
from pipelines.qse_spectra.method_completion import (
    ROUTE_SELECTED_QSE_FIXED_POOL,
    SELECTED_QSE_TERMINAL_STATUSES,
    STATUS_TARGET_WINDOW_CERTIFIED,
    classify_selected_qse_completion,
)
from pipelines.qse_spectra.paper_iii_problem import load_problem
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)
from pipelines.qse_spectra.static_campaign import canonical_digest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_selected_qse_fixed_pool_v1/failure_evidence.json"
)
EVIDENCE_SCHEMA_VERSION = "paper_iii_selected_qse_fixed_pool_failure_evidence_v2"
TARGET_ROOT_COUNT = 6
METHOD_ID = "record_first_adaptive_qse"
SELECTION_POLICY = {
    "mode": "geometry_selected",
    "max_records": "entire_fixed_pool",
    "geometry_target_roots": TARGET_ROOT_COUNT,
    "geometry_cost_discount_alpha": 1.0,
    "geometry_residual_stop": RESIDUAL_MARKER,
    "basis_vector_policy": {
        "reference_projection": "q0",
        "basis_vector_normalization": "raw_projected",
    },
    "compiled_cost_oracle": "marrakesh_graph_span_v1",
    "compiled_cost_weights": "two_qubit_only_v1",
}
RELEVANT_SOURCE_PATHS = (
    "pipelines/exact_bench/paper_iii_fixed_pool_failure_evidence.py",
    "pipelines/exact_bench/paper_iii_growth_trace_campaign.py",
    "pipelines/qse_spectra/compiled_costs.py",
    "pipelines/qse_spectra/core.py",
    "pipelines/qse_spectra/exchange_maintenance.py",
    "pipelines/qse_spectra/method_completion.py",
    "pipelines/qse_spectra/paper_iii_problem.py",
    "pipelines/qse_spectra/record_selection.py",
)


def assert_native_fixed_pool_trace(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject traces that contain another construction family."""

    if not rows:
        raise ValueError("native fixed-pool evidence requires at least one trace row.")
    for index, row in enumerate(rows):
        phase = str(row.get("phase", ""))
        if phase != "records":
            raise ValueError(
                f"native fixed-pool trace row {index} has phase {phase!r}; "
                "synthesis is allowed only in the separately declared hybrid route."
            )


def summarize_terminal_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count terminal outcomes with an explicit failure denominator."""

    counts = {status: 0 for status in SELECTED_QSE_TERMINAL_STATUSES}
    for record in records:
        completion = record.get("algorithm_completion")
        if not isinstance(completion, Mapping):
            raise ValueError("evidence record is missing algorithm_completion.")
        status = str(completion.get("terminal_status", ""))
        if status not in counts:
            raise ValueError(f"unknown selected-QSE terminal status {status!r}.")
        counts[status] += 1
    counts = {status: count for status, count in counts.items() if count}
    denominator = len(records)
    certified = counts.get(STATUS_TARGET_WINDOW_CERTIFIED, 0)
    uncertified = denominator - certified
    return {
        "cell_count": denominator,
        "certified_cell_count": certified,
        "uncertified_cell_count": uncertified,
        "uncertified_fraction": (
            None if denominator == 0 else float(uncertified) / float(denominator)
        ),
        "terminal_status_counts": dict(sorted(counts.items())),
    }


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relevant_source_provenance() -> dict[str, Any]:
    """Pin the small source surface that can change native outcomes."""

    hashes = {
        relative: _sha256_file(REPO_ROOT / relative)
        for relative in RELEVANT_SOURCE_PATHS
    }
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *RELEVANT_SOURCE_PATHS],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status_lines = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "source_revision": _git_revision(),
        "relevant_path_sha256": hashes,
        "relevant_source_tree_clean": not status_lines,
        "relevant_source_status": status_lines,
    }


def validate_failure_evidence(
    payload: Mapping[str, Any], *, require_clean_sources: bool = False
) -> None:
    """Fail closed on route mixing, denominator drift, or weak provenance."""

    if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unexpected fixed-pool failure-evidence schema.")
    if payload.get("method_id") != METHOD_ID:
        raise ValueError("failure evidence has the wrong method id.")
    if payload.get("completion_route") != ROUTE_SELECTED_QSE_FIXED_POOL:
        raise ValueError("failure evidence is not the fixed-pool completion route.")
    if payload.get("davidson_invoked") is not False:
        raise ValueError("native fixed-pool evidence cannot invoke Davidson.")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("failure evidence requires a nonempty records list.")
    regimes = [str(record.get("regime")) for record in records]
    if len(regimes) != len(set(regimes)):
        raise ValueError("failure evidence contains duplicate regimes.")
    if int(payload.get("expected_cell_count", -1)) != len(records):
        raise ValueError("expected_cell_count does not match acquired records.")
    if set(payload.get("acquired_regimes") or ()) != set(regimes):
        raise ValueError("acquired_regimes does not match evidence records.")
    if payload.get("summary") != summarize_terminal_records(records):
        raise ValueError("failure-evidence terminal summary does not recompute.")
    policy = payload.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("failure evidence is missing selection_policy.")
    if dict(policy) != SELECTION_POLICY:
        raise ValueError("failure evidence does not use the locked selection policy.")
    if payload.get("selection_policy_digest") != canonical_digest(dict(policy)):
        raise ValueError("selection_policy_digest does not match selection_policy.")
    if int(payload.get("target_root_count", -1)) != TARGET_ROOT_COUNT:
        raise ValueError("failure evidence does not use the locked target-root count.")
    cap = int(payload["dimension_cap"])
    stride = int(payload["stride"])
    for record in records:
        if int(record.get("dimension_cap", -1)) != cap:
            raise ValueError("per-record dimension cap drifted from the acquisition.")
        if int(record.get("stride", -1)) != stride:
            raise ValueError("per-record stride drifted from the acquisition.")
        rows = record.get("trace")
        if not isinstance(rows, list):
            raise ValueError("failure-evidence record is missing its trace.")
        assert_native_fixed_pool_trace(rows)
        completion = record.get("algorithm_completion")
        if not isinstance(completion, Mapping):
            raise ValueError("failure-evidence record is missing completion data.")
        if completion.get("completion_route") != ROUTE_SELECTED_QSE_FIXED_POOL:
            raise ValueError("record completion route is not fixed-pool selected QSE.")
        if completion.get("davidson_invoked") is not False:
            raise ValueError("record completion invoked Davidson.")
        if completion.get("terminal_status") not in SELECTED_QSE_TERMINAL_STATUSES:
            raise ValueError("record has an unknown terminal status.")
        traced = int(record["traced_terminal_k"])
        if traced != int(rows[-1]["k"]):
            raise ValueError("traced_terminal_k does not match the trace.")
        trace_support = [int(index) for index in rows[-1]["support_indices"]]
        trace_support_size = int(rows[-1]["support_size"])
        if trace_support_size != len(trace_support):
            raise ValueError("terminal trace row has an inconsistent support size.")
        terminal_support = [int(index) for index in record["terminal_support_indices"]]
        if terminal_support != trace_support:
            raise ValueError("terminal_support_indices does not match the terminal trace row.")
        if int(record.get("terminal_support_size", -1)) != trace_support_size:
            raise ValueError("terminal_support_size does not match the terminal trace row.")
        if int(completion["trace_terminal_record_count"]) != traced:
            raise ValueError("completion trace count does not match traced_terminal_k.")
        if int(completion["selector_terminal_record_count"]) != int(
            record["selected_record_count"]
        ):
            raise ValueError("completion selector count does not match selection.")
        certificate = completion.get("terminal_support_certificate")
        if not isinstance(certificate, Mapping):
            raise ValueError("completion is missing the terminal-support certificate.")
        if [int(index) for index in certificate.get("support_indices", ())] != terminal_support:
            raise ValueError("terminal-support certificate is not bound to the trace support.")
        selector_receipt = record.get("selector_terminal_receipt")
        if not isinstance(selector_receipt, Mapping):
            raise ValueError("record is missing its selector terminal receipt.")
        selected_count = int(record["selected_record_count"])
        reached_selector_terminal = bool(
            traced == selected_count and selected_count <= cap
        )
        recomputed = classify_selected_qse_completion(
            selector_receipt,
            trace_terminal_record_count=traced,
            selector_terminal_record_count=selected_count,
            target_root_count=int(payload["target_root_count"]),
            trace_reached_selector_terminal=reached_selector_terminal,
            terminal_support_certificate=certificate,
        )
        expected_completion = {
            "method_id": METHOD_ID,
            **recomputed,
            "record_stop": dict(selector_receipt),
        }
        if dict(completion) != expected_completion:
            raise ValueError(
                "algorithm_completion does not recompute from its receipt and "
                "terminal-support certificate."
            )
    provenance = payload.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("failure evidence is missing source provenance.")
    hashes = provenance.get("relevant_path_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(RELEVANT_SOURCE_PATHS):
        raise ValueError("failure evidence does not pin every relevant source path.")
    if require_clean_sources and provenance.get("relevant_source_tree_clean") is not True:
        raise ValueError("relevant source paths were dirty during evidence acquisition.")


def acquire_failure_evidence(
    *,
    regime_set: str,
    dimension_cap: int,
    stride: int,
    regimes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the fixed-pool route for a declared HH regime set."""

    if regime_set not in REGIME_SETS:
        raise ValueError(f"unknown regime set {regime_set!r}.")
    declared_regimes = tuple(str(row[0]) for row in REGIME_SETS[regime_set])
    requested_regimes = (
        declared_regimes if regimes is None else tuple(str(regime) for regime in regimes)
    )
    if len(requested_regimes) != len(set(requested_regimes)):
        raise ValueError("requested regimes contain duplicates.")
    unknown = sorted(set(requested_regimes) - set(declared_regimes))
    if unknown:
        raise ValueError(
            f"requested regimes are not in {regime_set!r}: {unknown}."
        )
    if not requested_regimes:
        raise ValueError("at least one regime must be requested.")
    source_start = relevant_source_provenance()
    records: list[dict[str, Any]] = []
    for regime, u, g_ep, n_ph_max in REGIME_SETS[regime_set]:
        if regime not in set(requested_regimes):
            continue
        started = time.perf_counter()
        problem = load_problem(
            regime=regime,
            u=u,
            g_ep=g_ep,
            n_ph_max=n_ph_max,
            target_roots=TARGET_ROOT_COUNT,
        )
        selection = select_static_qse_records(
            problem.basis,
            config=StaticRecordSelectionConfig(
                mode="geometry_selected",
                max_records=len(problem.basis),
                geometry_target_roots=TARGET_ROOT_COUNT,
                geometry_cost_discount_alpha=1.0,
                geometry_residual_stop=RESIDUAL_MARKER,
            ),
            hamiltonian=problem.hamiltonian,
            prepared_state=problem.ground,
            basis_vector_policy=_Q0_POLICY,
            compiled_costs=problem.costs,
        )
        order = [int(index) for index in selection.selected_original_indices]
        record_stop = dict(selection.geometry_stop or {"stop_reason": "missing"})
        rows, completion = trace_selected_qse_fixed_pool(
            problem,
            order,
            k_max=int(dimension_cap),
            stride=int(stride),
            record_stop=record_stop,
            target_root_count=TARGET_ROOT_COUNT,
        )
        assert_native_fixed_pool_trace(rows)
        terminal = rows[-1]
        finite = [
            float(row["max_root_abs_error"])
            for row in rows
            if row.get("max_root_abs_error") is not None
        ]
        record = {
            "regime": str(regime),
            "parameters": {
                "u": float(u),
                "g_ep": float(g_ep),
                "n_ph_max": int(n_ph_max),
            },
            "dimension_cap": int(dimension_cap),
            "stride": int(stride),
            "pool_size": len(problem.basis),
            "pool_digest": str(problem.pool_digest),
            "compiled_cost_vector_sha256": canonical_digest(
                [float(value) for value in problem.costs]
            ),
            "compiled_resource_vector_sha256": canonical_digest(
                [dict(row) for row in problem.resources]
            ),
            "reference_key": str(problem.reference_key),
            "selected_record_count": len(order),
            "traced_terminal_k": int(terminal["k"]),
            "terminal_support_size": int(terminal["support_size"]),
            "terminal_support_indices": [
                int(index) for index in terminal["support_indices"]
            ],
            "algorithm_completion": completion,
            "selector_terminal_receipt": record_stop,
            "post_run_exact_audit": {
                "feeds_online_decisions": False,
                "terminal_max_root_abs_error": terminal.get("max_root_abs_error"),
                "terminal_per_root_abs_error": terminal.get("per_root_abs_error"),
                "best_attained_max_root_abs_error": min(finite) if finite else None,
                "reference_excitations": [float(value) for value in problem.references],
            },
            "trace": rows,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
        records.append(record)
        print(
            f"{regime}: {completion['terminal_status']} "
            f"selected={len(order)} traced_k={terminal['k']}",
            flush=True,
        )

    records.sort(key=lambda row: str(row["regime"]))
    if not records:
        raise ValueError("requested regime filter acquired no evidence cells.")
    source_end = relevant_source_provenance()
    if source_start["relevant_path_sha256"] != source_end["relevant_path_sha256"]:
        raise RuntimeError("relevant source changed during evidence acquisition.")
    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "source_revision": str(source_end["source_revision"]),
        "source_provenance": {
            **source_end,
            "acquisition_start_revision": str(source_start["source_revision"]),
            "relevant_sources_unchanged_during_run": True,
        },
        "regime_set": str(regime_set),
        "requested_regimes": list(requested_regimes),
        "acquired_regimes": [str(record["regime"]) for record in records],
        "expected_cell_count": len(requested_regimes),
        "dimension_cap": int(dimension_cap),
        "stride": int(stride),
        "target_root_count": TARGET_ROOT_COUNT,
        "method_id": METHOD_ID,
        "completion_route": ROUTE_SELECTED_QSE_FIXED_POOL,
        "davidson_invoked": False,
        "selection_policy": dict(SELECTION_POLICY),
        "selection_policy_digest": canonical_digest(SELECTION_POLICY),
        "summary": summarize_terminal_records(records),
        "records": records,
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
    }
    validate_failure_evidence(payload, require_clean_sources=False)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime-set", choices=sorted(REGIME_SETS), required=True)
    parser.add_argument("--k-max", type=int, default=DEFAULT_DIMENSION_CAP)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--regimes", default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    wanted = (
        None
        if args.regimes is None
        else tuple(item.strip() for item in str(args.regimes).split(",") if item.strip())
    )
    payload = acquire_failure_evidence(
        regime_set=str(args.regime_set),
        dimension_cap=int(args.k_max),
        stride=int(args.stride),
        regimes=wanted,
    )
    validate_failure_evidence(payload, require_clean_sources=True)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output_json: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
