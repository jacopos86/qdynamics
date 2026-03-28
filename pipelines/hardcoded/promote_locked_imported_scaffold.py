#!/usr/bin/env python3
"""Promote an imported ADAPT artifact into locked fixed-scaffold format."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.adapt_circuit_cost import _load_adapt_result, reconstruct_imported_adapt_circuit
from src.quantum.ansatz_parameterization import deserialize_layout, project_runtime_theta_block_mean, serialize_layout


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_runtime_term_labels(parameterization: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    blocks = parameterization.get("blocks", [])
    if not isinstance(blocks, Sequence):
        raise ValueError("parameterization.blocks must be a sequence.")
    for block in blocks:
        if not isinstance(block, Mapping):
            raise ValueError("parameterization.blocks entries must be mappings.")
        terms = block.get("runtime_terms_exyz", [])
        if not isinstance(terms, Sequence):
            raise ValueError("parameterization block runtime_terms_exyz must be a sequence.")
        for term in terms:
            if not isinstance(term, Mapping):
                raise ValueError("runtime term entries must be mappings.")
            label = str(term.get("pauli_exyz", "")).strip()
            if label == "":
                raise ValueError("runtime term entry is missing pauli_exyz.")
            labels.append(label)
    if not labels:
        raise ValueError("parameterization does not contain any runtime_terms_exyz labels.")
    return labels


def build_locked_imported_scaffold_payload(
    source_payload: Mapping[str, Any],
    *,
    source_artifact_json: str | Path,
    subject_kind: str,
    term_order_id: str = "source_order_runtime",
    term_order_basis: str = "imported_runtime_parameterization",
    source_order_runtime_indices: Sequence[int] | None = None,
    recommended_backend_name: str = "ibm_marrakesh",
    recommended_optimization_level: int = 1,
    recommended_seed_transpiler: int = 7,
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(source_payload))
    settings = payload.get("settings", {})
    adapt_vqe = payload.get("adapt_vqe", {})
    if not isinstance(settings, Mapping):
        raise ValueError("source payload is missing settings mapping.")
    if not isinstance(adapt_vqe, Mapping):
        raise ValueError("source payload is missing adapt_vqe mapping.")

    parameterization = adapt_vqe.get("parameterization", None)
    if not isinstance(parameterization, Mapping):
        raise ValueError("source adapt_vqe must include serialized parameterization for locked scaffold promotion.")
    layout = deserialize_layout(parameterization)
    theta_runtime = np.asarray(adapt_vqe.get("optimal_point", []), dtype=float).reshape(-1)
    if int(theta_runtime.size) != int(layout.runtime_parameter_count):
        raise ValueError(
            f"runtime theta length mismatch for locked scaffold promotion: got {theta_runtime.size}, "
            f"expected {layout.runtime_parameter_count}"
        )
    raw_logical = np.asarray(adapt_vqe.get("logical_optimal_point", []), dtype=float).reshape(-1)
    if int(raw_logical.size) == int(layout.logical_parameter_count):
        theta_logical = raw_logical
    else:
        theta_logical = project_runtime_theta_block_mean(theta_runtime, layout)

    runtime_labels = _extract_runtime_term_labels(parameterization)
    if source_order_runtime_indices is None:
        source_indices = list(range(len(runtime_labels)))
    else:
        source_indices = [int(x) for x in source_order_runtime_indices]
    if len(source_indices) != len(runtime_labels):
        raise ValueError(
            "source_order_runtime_indices length must match the runtime term count "
            f"({len(runtime_labels)})."
        )

    settings_out = dict(settings)
    source_pool_type = (
        str(settings_out.get("adapt_pool")).strip()
        if settings_out.get("adapt_pool", None) not in {None, ""}
        else (
            str(adapt_vqe.get("pool_type")).strip()
            if adapt_vqe.get("pool_type", None) not in {None, ""}
            else None
        )
    )
    settings_out["adapt_pool"] = "fixed_scaffold_locked"

    adapt_out = dict(adapt_vqe)
    adapt_out["pool_type"] = "fixed_scaffold_locked"
    adapt_out["structure_locked"] = True
    adapt_out["fixed_scaffold_kind"] = str(subject_kind)
    adapt_out["parameterization"] = serialize_layout(layout)
    adapt_out["num_parameters"] = int(layout.runtime_parameter_count)
    adapt_out["logical_num_parameters"] = int(layout.logical_parameter_count)
    adapt_out["ansatz_depth"] = int(len(layout.blocks))
    adapt_out["optimal_point"] = [float(x) for x in theta_runtime.tolist()]
    adapt_out["logical_optimal_point"] = [float(x) for x in theta_logical.tolist()]
    adapt_out["fixed_scaffold_metadata"] = {
        "schema_version": 1,
        "route_family": "locked_imported_scaffold_v1",
        "subject_kind": str(subject_kind),
        "structure_locked": True,
        "operator_count": int(len(layout.blocks)),
        "runtime_term_count": int(layout.runtime_parameter_count),
        "term_order_id": str(term_order_id),
        "term_order_basis": str(term_order_basis),
        "source_order_runtime_indices": [int(x) for x in source_indices],
        "source_order_runtime_term_labels_exyz": list(runtime_labels),
        "runtime_term_labels_exyz": list(runtime_labels),
        "source_artifact_json": str(source_artifact_json),
        "source_pool_type": source_pool_type,
        "compile_recommendation": {
            "backend_name": str(recommended_backend_name),
            "optimization_level": int(recommended_optimization_level),
            "seed_transpiler": int(recommended_seed_transpiler),
        },
    }

    payload["generated_utc"] = _now_utc()
    payload["pipeline"] = "hh_promote_locked_imported_scaffold_v1"
    payload["source_artifact_json"] = str(source_artifact_json)
    payload["settings"] = settings_out
    payload["adapt_vqe"] = adapt_out
    return payload


def promote_locked_imported_scaffold(
    *,
    source_json: str | Path,
    output_json: str | Path,
    subject_kind: str,
    term_order_id: str = "source_order_runtime",
    term_order_basis: str = "imported_runtime_parameterization",
    source_order_runtime_indices: Sequence[int] | None = None,
    recommended_backend_name: str = "ibm_marrakesh",
    recommended_optimization_level: int = 1,
    recommended_seed_transpiler: int = 7,
) -> dict[str, Any]:
    source_path = Path(source_json)
    raw_payload = _load_adapt_result(source_path)
    bundle = reconstruct_imported_adapt_circuit(raw_payload)
    ansatz_meta = dict(bundle.get("ansatz_input_state_meta", {}))
    if not bool(ansatz_meta.get("available", False)):
        raise ValueError(
            f"locked scaffold promotion requires ansatz_input_state provenance; got {ansatz_meta.get('reason')!r}"
        )
    promoted = build_locked_imported_scaffold_payload(
        bundle["payload"],
        source_artifact_json=source_path,
        subject_kind=str(subject_kind),
        term_order_id=str(term_order_id),
        term_order_basis=str(term_order_basis),
        source_order_runtime_indices=source_order_runtime_indices,
        recommended_backend_name=str(recommended_backend_name),
        recommended_optimization_level=int(recommended_optimization_level),
        recommended_seed_transpiler=int(recommended_seed_transpiler),
    )
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(promoted, indent=2) + "\n", encoding="utf-8")
    return promoted


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--subject-kind", default="hh_promoted_locked_scaffold_v1")
    p.add_argument("--term-order-id", default="source_order_runtime")
    p.add_argument("--term-order-basis", default="imported_runtime_parameterization")
    p.add_argument("--source-order-runtime-indices", default=None)
    p.add_argument("--recommended-backend-name", default="ibm_marrakesh")
    p.add_argument("--recommended-optimization-level", type=int, default=1)
    p.add_argument("--recommended-seed-transpiler", type=int, default=7)
    return p


def _parse_indices(raw: str | None) -> list[int] | None:
    if raw in {None, ""}:
        return None
    return [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    promoted = promote_locked_imported_scaffold(
        source_json=str(args.source_json),
        output_json=str(args.output_json),
        subject_kind=str(args.subject_kind),
        term_order_id=str(args.term_order_id),
        term_order_basis=str(args.term_order_basis),
        source_order_runtime_indices=_parse_indices(args.source_order_runtime_indices),
        recommended_backend_name=str(args.recommended_backend_name),
        recommended_optimization_level=int(args.recommended_optimization_level),
        recommended_seed_transpiler=int(args.recommended_seed_transpiler),
    )
    print(f"output_json={Path(args.output_json)}")
    print(
        "runtime_term_count="
        f"{promoted['adapt_vqe']['fixed_scaffold_metadata']['runtime_term_count']}"
    )
    print(
        "subject_kind="
        f"{promoted['adapt_vqe']['fixed_scaffold_metadata']['subject_kind']}"
    )


if __name__ == "__main__":
    main()
