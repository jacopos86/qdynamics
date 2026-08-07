#!/usr/bin/env python3
"""Compile the terminal AP-McLachlan ansatz represented by a trajectory JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.drive_aligned import (
    augment_state_with_drive_aligned_generator,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    runtime_coordinate_labels,
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.support_atoms import (
    active_support_atoms,
    append_occurrence_base_label,
    candidate_append_atoms,
)
from pipelines.time_dynamics.diagnostics.avqds_results_report import (
    DEFAULT_BACKEND,
    compile_terminal_qiskit_cost,
)
from pipelines.time_dynamics.normalized_pauli_pool import (
    build_normalized_pauli_pool,
    runtime_input_with_normalized_candidate_pool,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    _load_runtime_input_or_raise,
)
from src.quantum.ansatz_parameterization import build_parameter_layout


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"AP trajectory must be a JSON object: {path}")
    return dict(payload)


def _drive_config(payload: Mapping[str, Any], runtime_input: Any) -> Any:
    profile = dict(dict(payload["hamiltonian"])["drive_profile"])
    request = runtime_input.resolved_problem.request
    return SimpleNamespace(
        enabled=True,
        n_sites=int(request.num_sites),
        ordering=str(request.ordering),
        drive_A=float(profile["A"]),
        drive_omega=float(profile["omega"]),
        drive_tbar=float(profile["tbar"]),
        drive_phi=float(profile["phi"]),
        drive_pattern=str(profile["pattern"]),
        drive_custom_weights=profile.get("custom_weights"),
        drive_include_identity=bool(profile.get("include_identity", False)),
        drive_time_sampling=str(profile.get("time_sampling", "midpoint")),
        drive_t0=float(profile.get("t0", 0.0)),
    )


def reconstruct_terminal_ap_compile_input(
    trajectory_path: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    payload = _load_json(trajectory_path)
    trajectory = dict(payload.get("trajectory", {}))
    metadata = dict(trajectory.get("metadata", {}))
    final_state = dict(payload.get("final_state", {}))
    source_path = Path(str(payload["source_artifact_json"]))
    runtime_input = _load_runtime_input_or_raise(
        source_path,
        loader_mode=metadata.get("loader_mode"),
        tag=metadata.get("tag"),
        generator_family=str(metadata.get("generator_family", "match_adapt")),
        fallback_family=str(metadata.get("fallback_family", "full_meta")),
        replay_candidate_pool_mode=metadata.get("replay_candidate_pool_mode"),
    )
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(
        runtime_input,
        drive_config=_drive_config(payload, runtime_input),
    )
    saved_normalized_pool = payload.get("normalized_candidate_pool")
    if isinstance(saved_normalized_pool, Mapping):
        profile = str(saved_normalized_pool.get("profile", "")).strip()
        if not profile:
            raise ValueError(
                "AP trajectory normalized_candidate_pool is missing its profile."
            )
        normalized_pool = build_normalized_pauli_pool(
            profile=profile,
            static_poly=hamiltonian.static_poly,
            drive_poly=hamiltonian.drive_poly,
            candidate_pool_terms=tuple(
                getattr(runtime_input, "candidate_pool_terms", ()) or ()
            ),
        )
        rebuilt_pool = normalized_pool.to_json_dict()
        expected_count = int(saved_normalized_pool.get("atom_count", -1))
        expected_digest = str(
            saved_normalized_pool.get("ordered_atom_contract_sha256", "")
        )
        if (
            int(rebuilt_pool["atom_count"]) != expected_count
            or str(rebuilt_pool["ordered_atom_contract_sha256"]) != expected_digest
        ):
            raise ValueError(
                "AP terminal normalized candidate-pool parity failed: "
                f"saved_count={expected_count}, rebuilt_count={rebuilt_pool['atom_count']}, "
                f"saved_digest={expected_digest!r}, "
                f"rebuilt_digest={rebuilt_pool['ordered_atom_contract_sha256']!r}."
            )
        runtime_input = runtime_input_with_normalized_candidate_pool(
            runtime_input,
            normalized_pool,
        )
    state = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode=str(final_state["parameterization_mode"]),
    )
    augmentation = augment_state_with_drive_aligned_generator(
        state,
        hamiltonian=hamiltonian,
        enabled=bool(dict(payload.get("drive_aligned_ansatz", {})).get("applied", True)),
    )
    state = augmentation.state
    term_lookup = {
        str(getattr(term, "label", "")): term
        for term in tuple(state.terms) + tuple(state.candidate_pool_terms)
    }
    term_lookup.update(
        {atom.atom_label: atom.term for atom in active_support_atoms(state)}
    )
    term_lookup.update(
        {
            atom.atom_label: atom.term
            for atom in candidate_append_atoms(
                state,
                allow_incomplete_candidate_pool=True,
            )
        }
    )
    selected_labels = tuple(str(value) for value in final_state["selected_term_labels"])
    for label in selected_labels:
        if label in term_lookup:
            continue
        base_label = append_occurrence_base_label(label)
        base_term = term_lookup.get(base_label)
        if base_term is not None:
            term_lookup[label] = replace(base_term, label=label)
    missing = tuple(label for label in selected_labels if label not in term_lookup)
    if missing:
        raise ValueError(
            "Cannot reconstruct terminal AP support; unresolved selected labels: "
            f"{missing[:8]}"
        )
    terms = tuple(term_lookup[label] for label in selected_labels)
    layout = build_parameter_layout(
        terms,
        ignore_identity=bool(state.layout.ignore_identity),
        coefficient_tolerance=float(state.layout.coefficient_tolerance),
        sort_terms=(str(state.layout.term_order).strip().lower() == "sorted"),
    )
    points = trajectory.get("points")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or not points:
        raise ValueError("AP trajectory is missing terminal trajectory point data.")
    terminal = dict(points[-1])
    theta = np.asarray(terminal["theta_runtime"], dtype=float).reshape(-1)
    expected_runtime_labels = tuple(
        str(value) for value in final_state["runtime_coordinate_labels"]
    )
    rebuilt_runtime_labels = runtime_coordinate_labels(
        layout,
        parameterization_mode=str(final_state["parameterization_mode"]),
    )
    parity = {
        "schema": "ap_terminal_support_reconstruction_parity_v1",
        "passed": bool(
            rebuilt_runtime_labels == expected_runtime_labels
            and int(layout.logical_parameter_count) == int(final_state["logical_parameter_count"])
            and int(layout.runtime_parameter_count) == int(final_state["runtime_parameter_count"])
            and int(theta.size) == int(final_state["runtime_parameter_count"])
        ),
        "selected_term_labels_match": True,
        "runtime_coordinate_labels_match": bool(
            rebuilt_runtime_labels == expected_runtime_labels
        ),
        "logical_parameter_count_rebuilt": int(layout.logical_parameter_count),
        "logical_parameter_count_saved": int(final_state["logical_parameter_count"]),
        "runtime_parameter_count_rebuilt": int(layout.runtime_parameter_count),
        "runtime_parameter_count_saved": int(final_state["runtime_parameter_count"]),
        "theta_length": int(theta.size),
    }
    if not parity["passed"]:
        raise ValueError(f"Terminal AP support reconstruction parity failed: {parity}")
    compile_input = SimpleNamespace(
        runtime_input=runtime_input,
        layout=layout,
        theta_runtime=theta,
        psi_ref=np.asarray(state.psi_ref, dtype=complex).reshape(-1),
        parity=parity,
        drive_aligned_ansatz=augmentation.to_json_dict(),
        diagnostic_redundancy_stress=dict(
            payload.get("diagnostic_redundancy_stress", {}) or {}
        ),
    )
    return compile_input, parity, payload


def build_ap_terminal_cost_table(
    *,
    trajectory_path: Path,
    label: str,
    backend_name: str,
    seed_transpiler: int,
    optimization_level: int,
) -> dict[str, Any]:
    compile_input, parity, payload = reconstruct_terminal_ap_compile_input(
        trajectory_path
    )
    result = compile_terminal_qiskit_cost(
        compile_input,
        backend_name=backend_name,
        seed_transpiler=seed_transpiler,
        optimization_level=optimization_level,
    )
    result["schema"] = "ap_terminal_qiskit_compile_v1"
    result["compile_scope"] = "final_active_ap_mclachlan_ansatz_at_terminal_time"
    summary = dict(payload.get("summary", {}))
    row = {
        "label": str(label),
        "trajectory_json": str(trajectory_path),
        "logical_parameter_count": int(summary["logical_parameter_count_final"]),
        "runtime_parameter_count": int(summary["runtime_parameter_count_final"]),
        "accepted_append_count": int(summary.get("accepted_append_count", 0)),
        "accepted_appended_coordinate_count": int(
            summary.get("accepted_appended_coordinate_count", 0)
        ),
        "accepted_delete_count": int(summary.get("accepted_delete_count", 0)),
        "accepted_deleted_coordinate_count": int(
            summary.get("accepted_deleted_coordinate_count", 0)
        ),
        "final_abs_energy_error": float(summary["final_abs_energy_error"]),
        "final_abs_doublon_error": float(summary["final_abs_doublon_error"]),
        "N2q": int(result["N2q"]),
        "D2q": int(result["D2q"]),
        "Dc": int(result["Dc"]),
        "qiskit_cost_status": "ok",
        "qiskit_cost_source": "final_active_ap_mclachlan_ansatz_qiskit_compile",
        "terminal_reconstruction_parity": parity,
    }
    return {
        "schema": "ap_terminal_qiskit_cost_table_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "compile_defaults": {
            "backend_name": str(result["backend_name"]),
            "requested_backend_name": str(result["requested_backend_name"]),
            "seed_transpiler": int(result["seed_transpiler"]),
            "optimization_level": int(result["optimization_level"]),
            "local_fake_only": True,
        },
        "rows": [row],
        "compile_result": result,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--output-cost-table-json", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--backend-name", default=DEFAULT_BACKEND)
    parser.add_argument("--seed-transpiler", type=int, default=7)
    parser.add_argument("--optimization-level", type=int, default=2)
    args = parser.parse_args(argv)
    payload = build_ap_terminal_cost_table(
        trajectory_path=args.trajectory_json,
        label=str(args.label),
        backend_name=str(args.backend_name),
        seed_transpiler=int(args.seed_transpiler),
        optimization_level=int(args.optimization_level),
    )
    output = Path(args.output_cost_table_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = dict(payload["compile_result"])
    print(
        json.dumps(
            {
                "output_cost_table_json": str(output.resolve()),
                "N2q": int(result["N2q"]),
                "D2q": int(result["D2q"]),
                "Dc": int(result["Dc"]),
                "terminal_reconstruction_parity": payload["rows"][0][
                    "terminal_reconstruction_parity"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
