#!/usr/bin/env python3
"""Compile every preserved weak-weak macro prefix under locked Qiskit conventions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pipelines.exact_bench.paper_i_hh_recovery_prefix_qiskit_sidecar import (
    compile_historical_displayed_convention,
    derive_execution_order_repaired_checkpoint,
    reconstruct_reference_state,
    validate_active_prefix_checkpoint,
)
from pipelines.hardcoded.adapt_circuit_execution import build_ansatz_circuit
from pipelines.qiskit_backend_tools import (
    compile_circuit_for_backend,
    compiled_gate_stats,
    load_local_fake_backend,
    safe_circuit_depth,
)


JQ_REDUCTION = r"""
{
  ansatz_input_state: .ansatz_input_state,
  history: [
    .adapt_vqe.history[] |
    {
      round: .depth,
      selected_op: .selected_op,
      selected_class: .physical_operator_hh_full_meta_class,
      selected_lane: .physical_operator_lane,
      accepted_drop: ([0, (-.delta_energy)] | max),
      checkpoint: .active_prefix_checkpoint
    }
  ]
}
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-final-sidecar-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def reduce_result(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        ["jq", "-c", JQ_REDUCTION, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def metric_delta(current: dict[str, int], previous: dict[str, int]) -> dict[str, int]:
    return {key: int(current[key]) - int(previous[key]) for key in ("N2q", "D2q", "Dc")}


def compile_current_metrics(validated: Any, reference_state: np.ndarray, backend: Any) -> dict[str, int]:
    circuit = build_ansatz_circuit(
        validated.layout,
        np.asarray(validated.runtime_parameters, dtype=float),
        int(validated.num_qubits),
        ref_state=np.asarray(reference_state, dtype=complex),
    )
    compiled_info = compile_circuit_for_backend(
        circuit,
        backend,
        seed_transpiler=7,
        optimization_level=1,
    )
    compiled_circuit = compiled_info["compiled"]
    stats = dict(compiled_gate_stats(compiled_circuit))
    metrics = {
        "N2q": int(stats["compiled_count_2q"]),
        "D2q": int(stats["compiled_depth_2q"]),
        "Dc": int(safe_circuit_depth(compiled_circuit)),
    }
    del compiled_circuit, compiled_info, circuit, stats
    return metrics


def main() -> None:
    args = parse_args()
    observed_sha256 = sha256_file(args.result_json)
    if observed_sha256 != args.expected_result_sha256:
        raise ValueError(
            f"result SHA-256 mismatch: expected={args.expected_result_sha256}, observed={observed_sha256}"
        )
    reduced = reduce_result(args.result_json)
    history = reduced["history"]
    if len(history) != 50:
        raise ValueError(f"expected 50 preserved prefix checkpoints, found {len(history)}")

    first_repaired, _first_repair = derive_execution_order_repaired_checkpoint(
        history[0]["checkpoint"], expected_outer_iteration=1
    )
    first_validated = validate_active_prefix_checkpoint(
        first_repaired, expected_outer_iteration=1
    )
    reference_state, reference_meta = reconstruct_reference_state(
        reduced, num_qubits=int(first_validated.num_qubits)
    )
    backend, resolved_backend_name = load_local_fake_backend("FakeMarrakesh")

    previous = {
        "historical_displayed": {"N2q": 0, "D2q": 0, "Dc": 0},
        "current_jr_fake_marrakesh": {"N2q": 0, "D2q": 0, "Dc": 0},
    }
    rows: list[dict[str, Any]] = []
    for expected_round, history_row in enumerate(history, start=1):
        observed_round = int(history_row["round"])
        if observed_round != expected_round:
            raise ValueError(
                f"noncontiguous preserved history: expected round {expected_round}, found {observed_round}"
            )
        repaired_checkpoint, repair_summary = derive_execution_order_repaired_checkpoint(
            history_row["checkpoint"], expected_outer_iteration=expected_round
        )
        validated = validate_active_prefix_checkpoint(
            repaired_checkpoint, expected_outer_iteration=expected_round
        )
        historical = compile_historical_displayed_convention(
            validated, reference_state=reference_state
        )
        metrics = {
            "historical_displayed": dict(historical["metrics"]),
            "current_jr_fake_marrakesh": compile_current_metrics(
                validated, reference_state, backend
            ),
        }
        deltas = {
            convention: metric_delta(metric_values, previous[convention])
            for convention, metric_values in metrics.items()
        }
        rows.append(
            {
                "round": expected_round,
                "selected_op": history_row["selected_op"],
                "selected_class": history_row["selected_class"],
                "selected_lane": history_row["selected_lane"],
                "accepted_drop": history_row["accepted_drop"],
                "checkpoint_sha256": validated.checkpoint_sha256,
                "checkpoint_execution_order_repair": repair_summary,
                "metrics": metrics,
                "marginal_prefix_delta": deltas,
            }
        )
        previous = metrics
        del historical, validated, repaired_checkpoint
        gc.collect()
        print(
            f"compiled prefix {expected_round:02d}/50: "
            f"table N2q={metrics['historical_displayed']['N2q']}, "
            f"current N2q={metrics['current_jr_fake_marrakesh']['N2q']}",
            flush=True,
        )

    final_sidecar = json.loads(args.expected_final_sidecar_json.read_text())
    expected_final = {
        "historical_displayed": final_sidecar["historical_displayed_convention"]["metrics"],
        "current_jr_fake_marrakesh": final_sidecar["current_jr_fake_marrakesh_convention"]["metrics"],
    }
    for convention, expected_metrics in expected_final.items():
        if rows[-1]["metrics"][convention] != expected_metrics:
            raise ValueError(
                f"final-prefix {convention} metrics disagree with locked sidecar: "
                f"recomputed={rows[-1]['metrics'][convention]}, expected={expected_metrics}"
            )

    negative_delta_counts = {
        convention: {
            metric: sum(
                int(row["marginal_prefix_delta"][convention][metric]) < 0 for row in rows
            )
            for metric in ("N2q", "D2q", "Dc")
        }
        for convention in previous
    }
    payload = {
        "schema": "paper_i_weak_weak_macro_prefix_qiskit_cost_curve_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "result_json": str(args.result_json.resolve()),
            "result_sha256": observed_sha256,
            "final_sidecar_json": str(args.expected_final_sidecar_json.resolve()),
            "final_sidecar_sha256": sha256_file(args.expected_final_sidecar_json),
        },
        "scope": {
            "regime": "weak_weak",
            "macro_only": True,
            "prefix_rounds": len(rows),
            "reference_state": reference_meta,
            "current_backend": str(resolved_backend_name),
        },
        "definition": {
            "marginal_prefix_delta": "Qiskit metric at independently compiled prefix k minus the same metric at prefix k-1; k=0 has zero two-qubit count and depth for the preserved HF basis reference",
            "warning": "independent whole-prefix transpilation can make marginal deltas nonmonotone; negative deltas are reported, never clipped",
        },
        "negative_delta_counts": negative_delta_counts,
        "final_prefix_lock": expected_final,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
