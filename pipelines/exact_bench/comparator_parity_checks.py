#!/usr/bin/env python3
"""Read-only artifact-level parity checks for exact-bench comparators.

These helpers compare already-produced row/result dictionaries.  They do not run
physics kernels, submit jobs, edit table-support files, or promote manuscript
evidence.  The intended use is a tiny smoke/parity pass that writes a
``comparator_parity_sidecar.json`` next to scratch artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.exact_bench.comparator_parity_sidecar import (
    build_comparator_parity_sidecar,
    write_comparator_parity_sidecar,
)

_ENERGY_KEYS = (
    "energy",
    "final_energy",
    "optimizer_reported_energy",
    "optimizer_decision_energy",
)
_SELECTED_SEQUENCE_KEYS = (
    "selected_operators",
    "selected_operator_labels",
    "selected_generators",
    "selected_generator_labels",
    "selected_operator_sequence",
    "qiskit_selected_operator_labels",
)
_COMPILED_COST_KEYS = (
    "compiled_count_2q_total",
    "compiled_depth_2q_total",
    "compiled_depth_total",
    "count_2q",
    "circuit_depth",
)
_STATE_INFidelity_KEYS = (
    "state_infidelity",
    "parity_state_infidelity",
    "final_state_infidelity",
    "statevector_infidelity",
)
_STATE_FIDELITY_KEYS = (
    "state_fidelity",
    "final_state_fidelity",
    "state_overlap_abs2",
)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _energy(row: Mapping[str, Any]) -> float | None:
    return _finite_float(_first_present(row, _ENERGY_KEYS))


def _selected_sequence(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    value = _first_present(row, _SELECTED_SEQUENCE_KEYS)
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    return None


def _compiled_cost(row: Mapping[str, Any]) -> dict[str, Any] | None:
    out = {key: row.get(key) for key in _COMPILED_COST_KEYS if key in row and row.get(key) is not None}
    return out or None


def _state_infidelity(row: Mapping[str, Any]) -> float | None:
    direct = _finite_float(_first_present(row, _STATE_INFidelity_KEYS))
    if direct is not None:
        return direct
    fidelity = _finite_float(_first_present(row, _STATE_FIDELITY_KEYS))
    if fidelity is None:
        return None
    return max(0.0, 1.0 - float(fidelity))


def _payload_row(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result")
    if isinstance(result, Mapping):
        return result
    rows = payload.get("rows")
    if isinstance(rows, Sequence) and rows and isinstance(rows[0], Mapping):
        return rows[0]
    return payload


def load_row_artifact(path: str | Path) -> Mapping[str, Any]:
    """Load a row-like mapping from result/manifest/rows JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return _payload_row(data)


def build_static_row_parity_sidecar(
    *,
    algorithm_id: str,
    subject_row: Mapping[str, Any],
    reference_row: Mapping[str, Any],
    parity_reference_algorithm_id: str,
    subject_artifact: str | Path | None = None,
    parity_reference_artifact: str | Path | None = None,
    energy_tolerance: float = 1.0e-8,
    state_infidelity_tolerance: float = 1.0e-8,
    parity_scope: str = "artifact_level_static_row_common_quantities",
    runner_module: str | None = "pipelines.exact_bench.comparator_parity_checks",
) -> dict[str, Any]:
    """Compare common row-level quantities and return a normalized sidecar.

    Missing method-specific quantities are reported as bounded gaps; they do not
    imply pass.  A sidecar is ``passed`` only when at least one common quantity is
    compared and none of the requested/common comparisons fails.
    """
    subject_energy = _energy(subject_row)
    reference_energy = _energy(reference_row)
    energy_delta = (
        None
        if subject_energy is None or reference_energy is None
        else abs(float(subject_energy) - float(reference_energy))
    )

    subject_sequence = _selected_sequence(subject_row)
    reference_sequence = _selected_sequence(reference_row)
    sequence_match = (
        None
        if subject_sequence is None or reference_sequence is None
        else subject_sequence == reference_sequence
    )

    subject_cost = _compiled_cost(subject_row)
    reference_cost = _compiled_cost(reference_row)
    compiled_cost_match = None if subject_cost is None or reference_cost is None else subject_cost == reference_cost

    subject_infidelity = _state_infidelity(subject_row)
    reference_infidelity = _state_infidelity(reference_row)
    state_infidelity_delta = (
        None
        if subject_infidelity is None or reference_infidelity is None
        else abs(float(subject_infidelity) - float(reference_infidelity))
    )

    comparisons = {
        "energy": {
            "subject": subject_energy,
            "reference": reference_energy,
            "abs_delta": energy_delta,
            "tolerance": float(energy_tolerance),
            "passed": None if energy_delta is None else bool(energy_delta <= float(energy_tolerance)),
        },
        "selected_sequence": {
            "subject": None if subject_sequence is None else list(subject_sequence),
            "reference": None if reference_sequence is None else list(reference_sequence),
            "passed": sequence_match,
        },
        "compiled_cost": {
            "subject": subject_cost,
            "reference": reference_cost,
            "passed": compiled_cost_match,
        },
        "state_infidelity": {
            "subject": subject_infidelity,
            "reference": reference_infidelity,
            "abs_delta": state_infidelity_delta,
            "tolerance": float(state_infidelity_tolerance),
            "passed": None
            if state_infidelity_delta is None
            else bool(state_infidelity_delta <= float(state_infidelity_tolerance)),
        },
    }
    failed = [name for name, item in comparisons.items() if item.get("passed") is False]
    compared = [name for name, item in comparisons.items() if item.get("passed") is not None]
    missing = [name for name, item in comparisons.items() if item.get("passed") is None]
    if failed:
        parity_status = "failed"
    elif compared and missing:
        parity_status = "partial_common_quantities_passed"
    elif compared:
        parity_status = "passed"
    else:
        parity_status = "not_run_no_common_quantities"

    return build_comparator_parity_sidecar(
        algorithm_id=algorithm_id,
        runner_module=runner_module,
        subject_artifact=subject_artifact,
        parity_status=parity_status,
        parity_scope=parity_scope,
        parity_reference_algorithm_id=parity_reference_algorithm_id,
        parity_reference_artifact=parity_reference_artifact,
        parity_energy_abs_delta=energy_delta,
        parity_state_infidelity=state_infidelity_delta,
        parity_selected_generators_match=sequence_match,
        parity_compiled_cost_match=compiled_cost_match,
        extra={
            "comparisons": comparisons,
            "compared_quantities": compared,
            "missing_quantities": missing,
            "failed_quantities": failed,
            "subject_algorithm_id": subject_row.get("algorithm_id") or subject_row.get("method_id"),
            "reference_algorithm_id": reference_row.get("algorithm_id") or reference_row.get("method_id"),
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, help="Subject row/result JSON artifact")
    parser.add_argument("--reference", required=True, help="Reference row/result JSON artifact")
    parser.add_argument("--algorithm-id", required=True, help="Subject comparator algorithm_id")
    parser.add_argument("--parity-reference-algorithm-id", required=True, help="Reference comparator algorithm_id")
    parser.add_argument("--output-dir", required=True, help="Directory for comparator_parity_sidecar.json")
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--state-infidelity-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--parity-scope", default="artifact_level_static_row_common_quantities")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    subject_row = load_row_artifact(args.subject)
    reference_row = load_row_artifact(args.reference)
    payload = build_static_row_parity_sidecar(
        algorithm_id=args.algorithm_id,
        subject_row=subject_row,
        reference_row=reference_row,
        parity_reference_algorithm_id=args.parity_reference_algorithm_id,
        subject_artifact=args.subject,
        parity_reference_artifact=args.reference,
        energy_tolerance=float(args.energy_tolerance),
        state_infidelity_tolerance=float(args.state_infidelity_tolerance),
        parity_scope=str(args.parity_scope),
    )
    write_comparator_parity_sidecar(args.output_dir, payload)
    print(json.dumps(payload, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_static_row_parity_sidecar",
    "load_row_artifact",
    "main",
]
