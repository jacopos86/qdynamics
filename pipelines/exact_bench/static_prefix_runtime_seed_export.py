#!/usr/bin/env python3
"""Export AP-loadable runtime seeds from arbitrary static-ADAPT prefixes.

This route is intentionally support-sequence driven.  It does not rerun ADAPT
selection.  It reads the selected batches recorded in a source static-ADAPT
artifact, replays those batches up to a requested active prefix depth, refits
the resulting fixed prefix support, and emits a normal Paper-II runtime seed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.exact_bench import generic_static_adapt_variants as variants
from pipelines.exact_bench.molecular_vibronic_h2_fixture_override import (
    with_molecular_vibronic_h2_fixture_override,
)
from pipelines.exact_bench.table_i_canonical_cases import table_i_canonical_spec_by_case_id
from src.quantum.compiled_polynomial import compile_polynomial_action, energy_via_one_apply


SCHEMA = "static_prefix_runtime_seed_export_v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}.")
    return dict(payload)


def _result_block(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result", payload)
    if not isinstance(result, Mapping):
        raise ValueError("source JSON has no result object.")
    return result


def _source_algorithm_id(payload: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    raw = (
        payload.get("algorithm_id")
        or payload.get("method_id")
        or result.get("algorithm_id")
        or result.get("method_id")
    )
    if raw in {None, ""}:
        raise ValueError("source JSON has no algorithm_id/method_id.")
    return str(raw)


def _source_case_id(payload: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    raw = payload.get("case_id") or result.get("case_id")
    if raw in {None, ""}:
        raise ValueError("source JSON has no case_id.")
    return str(raw)


def _source_family(payload: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    raw = payload.get("family") or result.get("family") or result.get("problem")
    if raw in {None, ""}:
        raise ValueError("source JSON has no family/problem key.")
    return str(raw)


def _selected_batches_from_history(
    payload: Mapping[str, Any],
    *,
    prefix_depth: int,
) -> tuple[list[list[str]], Mapping[str, Any]]:
    if int(prefix_depth) < 0:
        raise ValueError("prefix_depth must be nonnegative.")
    result = _result_block(payload)
    history = result.get("adapt_history", None)
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes, bytearray)):
        raise ValueError("source result has no adapt_history sequence.")
    selected_batches: list[list[str]] = []
    active_count = 0
    last_row: Mapping[str, Any] | None = None
    if int(prefix_depth) == 0:
        return selected_batches, {}
    for raw_row in history:
        if not isinstance(raw_row, Mapping):
            continue
        raw_labels = raw_row.get("selected_batch_labels", None)
        if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, (str, bytes, bytearray)):
            raise ValueError("adapt_history row missing selected_batch_labels.")
        labels = [str(label) for label in raw_labels]
        if not labels:
            continue
        next_count = int(active_count + len(labels))
        if next_count > int(prefix_depth):
            raise ValueError(
                "prefix_depth cuts through an adaptive batch: "
                f"active_count={active_count}, batch_size={len(labels)}, prefix_depth={prefix_depth}."
            )
        selected_batches.append(labels)
        active_count = next_count
        last_row = raw_row
        if active_count == int(prefix_depth):
            return selected_batches, dict(last_row)
    raise ValueError(
        f"source history ended at active depth {active_count}, before requested prefix_depth={prefix_depth}."
    )


def _candidate_batches_from_labels(
    *,
    pool: Sequence[Any],
    selected_batches: Sequence[Sequence[str]],
) -> list[list[Any]]:
    by_label = {str(candidate.label): candidate for candidate in pool}
    out: list[list[Any]] = []
    missing: list[str] = []
    for batch in selected_batches:
        resolved: list[Any] = []
        for label in batch:
            if str(label) not in by_label:
                missing.append(str(label))
                continue
            resolved.append(by_label[str(label)])
        out.append(resolved)
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"prefix selected label(s) absent from full_meta pool: {preview}")
    return out


def _candidate_batches_with_execution_mode(
    candidate_batches: Sequence[Sequence[Any]],
    execution_mode: str | None,
) -> list[list[Any]]:
    mode = None if execution_mode in {None, "", "current"} else str(execution_mode)
    if mode is None:
        return [list(batch) for batch in candidate_batches]
    if mode not in {"termwise_product", "grouped_exact"}:
        raise ValueError(
            "selected_execution_mode must be current, termwise_product, or grouped_exact."
        )
    return [
        [replace(candidate, execution_mode=mode) for candidate in batch]
        for batch in candidate_batches
    ]


def _reference_kwargs_from_source(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "same_cutoff_exact_gs_energy": result.get("same_cutoff_exact_gs_energy"),
        "exact_reference_energy": result.get("exact_reference_energy"),
        "exact_reference_n_ph_max": result.get("exact_reference_n_ph_max"),
        "primary_energy_metric": result.get("primary_energy_metric"),
        "same_cutoff_error_role": result.get("same_cutoff_error_role"),
    }


def run_static_prefix_runtime_seed_export(
    *,
    source_json: str | Path,
    prefix_depth: int,
    output_dir: str | Path,
    optimizer_kind: str = "powell",
    optimizer_maxiter: int = 200,
    table_i_suite_profile: str | None = None,
    pool_term_cap: int | None = None,
    selected_execution_mode: str | None = None,
) -> dict[str, Any]:
    source_path = Path(source_json)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started_utc = variants._utc_now()
    t0 = time.perf_counter()
    source_payload = _read_json(source_path)
    source_result = _result_block(source_payload)
    family = _source_family(source_payload, source_result)
    case_id = _source_case_id(source_payload, source_result)
    algorithm_id = _source_algorithm_id(source_payload, source_result)
    config = variants._get_config(algorithm_id)
    selected_label_batches, source_prefix_row = _selected_batches_from_history(
        source_payload,
        prefix_depth=int(prefix_depth),
    )
    spec = with_molecular_vibronic_h2_fixture_override(
        table_i_canonical_spec_by_case_id(family, case_id, table_i_suite_profile),
        family=family,
    )
    context = variants._resolve_context_from_spec(spec)
    psi_ref = np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(psi_ref))
    if norm <= 0.0:
        raise ValueError("reference state has zero norm.")
    psi_ref = psi_ref / norm
    pool = variants.build_full_meta_candidate_pool(context, max_terms=pool_term_cap)
    candidate_batches = _candidate_batches_with_execution_mode(
        _candidate_batches_from_labels(
            pool=pool,
            selected_batches=selected_label_batches,
        ),
        selected_execution_mode,
    )
    selected: list[Any] = []
    selected_batches: list[list[Any]] = []
    theta = np.zeros(0, dtype=float)
    optimizer_key = str(optimizer_kind).strip().lower()
    if optimizer_key not in {"powell", "bfgs", "rotosolve"}:
        raise ValueError("prefix runtime seed export currently supports optimizer_kind=powell, bfgs, or rotosolve.")
    if optimizer_key in {"powell", "bfgs"} and not variants.has_scipy_minimize_support():
        raise ImportError("scipy.optimize.minimize is required for prefix seed refit.")
    minimize_fn = variants._import_scipy_minimize() if optimizer_key in {"powell", "bfgs"} else None
    pauli_action_cache: dict[str, Any] = {}
    h_compiled = compile_polynomial_action(
        context.hamiltonian,
        tol=1e-12,
        pauli_action_cache=pauli_action_cache,
    )
    progress_rows: list[dict[str, Any]] = []
    for step_index, batch in enumerate(candidate_batches):
        selected.extend(batch)
        selected_batches.append(list(batch))
        theta = np.concatenate(
            [
                np.asarray(theta, dtype=float).reshape(-1),
                np.zeros(int(len(batch)), dtype=float),
            ]
        )
        theta, energy, info = variants._optimize_selected(
            minimize_fn=minimize_fn,
            selected=selected,
            x0=np.asarray(theta, dtype=float).reshape(-1),
            psi_ref=psi_ref,
            h_compiled=h_compiled,
            pauli_action_cache=pauli_action_cache,
            optimizer_maxiter=int(optimizer_maxiter),
            optimizer_method=optimizer_key,
            parameterization_mode="logical_shared",
            decision_scope={
                "static_prefix_runtime_seed_export": True,
                "source_json": str(source_path),
                "prefix_depth": int(prefix_depth),
                "prefix_step_index": int(step_index),
            },
        )
        progress_rows.append(
            {
                "step_index": int(step_index),
                "active_depth": int(len(selected)),
                "batch_labels": [str(candidate.label) for candidate in batch],
                "energy": float(energy),
                "theta_count": int(theta.size),
                "optimizer_info": dict(info),
            }
        )
        variants._write_json(output / "prefix_refit_progress.json", {"schema": SCHEMA, "rows": progress_rows})
    psi_final = variants._prepare_selected_state(
        selected=selected,
        theta=np.asarray(theta, dtype=float).reshape(-1),
        psi_ref=psi_ref,
        pauli_action_cache=pauli_action_cache,
        parameterization_mode="logical_shared",
    )
    energy_final, _ = energy_via_one_apply(psi_final, h_compiled)
    energy_final = float(energy_final)
    reference_metrics = variants._normalize_reference_metric_fields(
        **_reference_kwargs_from_source(source_result),
        fallback_same_cutoff_energy=variants._safe_exact_energy(context),
    )
    error_fields = variants._energy_error_fields(energy_final, reference_metrics)
    finished_utc = variants._utc_now()
    row = variants._base_row(
        family=family,
        case_id=case_id,
        config=config,
        status="ok",
        started_utc=started_utc,
        finished_utc=finished_utc,
    )
    row.update(
        {
            "energy": energy_final,
            "exact_energy": reference_metrics.get("same_cutoff_exact_gs_energy"),
            "exact_gs_energy": reference_metrics.get("same_cutoff_exact_gs_energy"),
            "same_cutoff_exact_gs_energy": reference_metrics.get("same_cutoff_exact_gs_energy"),
            "exact_reference_energy": reference_metrics.get("exact_reference_energy"),
            "exact_reference_n_ph_max": reference_metrics.get("exact_reference_n_ph_max"),
            "primary_energy_metric": reference_metrics.get("primary_energy_metric"),
            "primary_reference_source": reference_metrics.get("primary_reference_source"),
            "same_cutoff_error_role": reference_metrics.get("same_cutoff_error_role"),
            "adapt_stop_reason": "prefix_runtime_seed_export",
            "position_policy": "append",
            "position_optimized_geo_adapt": False,
            "parameterization_mode": "logical_shared",
            "theta_coordinate_mode": "logical_shared",
            "selected_operators": [str(candidate.label) for candidate in selected],
            "source_static_prefix_depth": int(prefix_depth),
            "source_static_prefix_row": dict(source_prefix_row),
            "source_static_prefix_execution_mode_override": (
                None
                if selected_execution_mode in {None, "", "current"}
                else str(selected_execution_mode)
            ),
            **error_fields,
        }
    )
    runtime_seed = variants._build_runtime_seed_payload(
        context=context,
        family=family,
        case_id=case_id,
        config=config,
        selected=tuple(selected),
        selected_batches=tuple(tuple(batch) for batch in selected_batches),
        theta=np.asarray(theta, dtype=float).reshape(-1),
        psi_ref=psi_ref,
        psi_final=psi_final,
        row=row,
        spec=spec,
        generated_utc=finished_utc,
    )
    runtime_seed["paper_ii_static_seed_export"]["runtime_loadability_status"] = (
        "prefix_refit_runtime_seed_sidecar_written_not_dry_loaded"
    )
    runtime_seed["paper_ii_static_seed_export"]["source_prefix_depth"] = int(prefix_depth)
    runtime_seed["paper_ii_static_seed_export"]["source_json"] = str(source_path)
    source_prefix_energy = variants._finite_float_or_none(source_prefix_row.get("energy_after"))
    source_prefix_abs_delta = variants._finite_float_or_none(
        source_prefix_row.get("abs_delta_e_same_cutoff_after", source_prefix_row.get("abs_delta_e_after"))
    )
    payload = {
        "schema": SCHEMA,
        "status": "completed",
        "generated_utc": finished_utc,
        "started_utc": started_utc,
        "runtime_s": float(time.perf_counter() - t0),
        "source_json": str(source_path),
        "family": family,
        "case_id": case_id,
        "algorithm_id": algorithm_id,
        "prefix_depth": int(prefix_depth),
        "optimizer_kind": optimizer_key,
        "optimizer_maxiter": int(optimizer_maxiter),
        "selected_execution_mode": (
            "current"
            if selected_execution_mode in {None, "", "current"}
            else str(selected_execution_mode)
        ),
        "pool_term_count": int(len(pool)),
        "selected_operator_count": int(len(selected)),
        "selected_operator_labels": [str(candidate.label) for candidate in selected],
        "selected_operator_batches": [[str(candidate.label) for candidate in batch] for batch in selected_batches],
        "energy": energy_final,
        "error_fields": dict(error_fields),
        "source_prefix_energy_after": source_prefix_energy,
        "source_prefix_abs_delta_e_after": source_prefix_abs_delta,
        "source_prefix_energy_delta": (
            None if source_prefix_energy is None else float(energy_final - float(source_prefix_energy))
        ),
        "progress_rows": progress_rows,
        "runtime_seed_json": str(output / "runtime_seed.json"),
        "runtime_seed_schema": runtime_seed.get("schema"),
        "guardrails": {
            "uses_exact_for_decision": False,
            "uses_reference_for_decision": False,
            "exact_reference_usage": "reporting_only_after_prefix_refit",
            "support_selection_source": "source_adapt_history_selected_batches",
            "optimizer_scope": "fixed_support_prefix_refit",
            "selected_execution_mode_source": (
                "current_pool"
                if selected_execution_mode in {None, "", "current"}
                else "explicit_source_compatibility_override"
            ),
        },
    }
    variants._write_json(output / "runtime_seed.json", runtime_seed)
    variants._write_json(output / "result.json", payload)
    variants._write_json(output / "static_prefix_runtime_seed_export.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--prefix-depth", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--optimizer-kind", default="powell")
    parser.add_argument("--optimizer-maxiter", type=int, default=200)
    parser.add_argument("--table-i-suite-profile", default=None)
    parser.add_argument("--pool-term-cap", type=int, default=None)
    parser.add_argument(
        "--selected-execution-mode",
        choices=("current", "termwise_product", "grouped_exact"),
        default="current",
    )
    args = parser.parse_args(argv)
    run_static_prefix_runtime_seed_export(
        source_json=args.source_json,
        prefix_depth=int(args.prefix_depth),
        output_dir=args.output_dir,
        optimizer_kind=str(args.optimizer_kind),
        optimizer_maxiter=int(args.optimizer_maxiter),
        table_i_suite_profile=args.table_i_suite_profile,
        pool_term_cap=args.pool_term_cap,
        selected_execution_mode=args.selected_execution_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
