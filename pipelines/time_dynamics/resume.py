"""Continuation of a completed trajectory from its final state.

Long horizons are reached by continuing, not by re-running: the expensive cells
of a production matrix take hours to reach their end time, and extending them
from t=0 discards that work. A continuation needs the propagated state, not the
original seed, so a completed run records everything required to rebuild its
final support exactly.

What must be preserved is the implemented unitary, and that is fixed by the
ordered list of runtime coordinates -- each a single Pauli rotation with its
word, coefficient, and register width -- together with the parameter vector.
The reference state is unchanged by propagation, so it comes from the original
seed artifact rather than being duplicated here.

Scope: continuation is physically faithful, not decision-identical. The
propagated state and its energy continue exactly -- a split trajectory agrees
with an uninterrupted one to solver tolerance -- but controller history
(deletion history, cooldowns, persistence counters) does not cross the
boundary, so the continued leg may make different, equally valid structural
choices than an uninterrupted run would. Serializing that history would make
continuation decision-identical and is the natural next step if a comparison
ever needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

RESUME_SCHEMA_V1 = "paper_ii_resume_state_v1"


def build_resume_state(
    state: Any,
    *,
    theta_runtime: Sequence[float],
    time: float,
    seed_artifact_json: str | None,
) -> dict[str, Any]:
    """Serialize the end of a trajectory so it can be continued exactly."""

    from pipelines.time_dynamics.ap_mclachlan.state import _per_coordinate_terms
    from src.quantum.ansatz_parameterization import iter_runtime_rotation_terms

    coordinates: list[dict[str, Any]] = []
    for label, term in zip(state.runtime_coordinate_labels, _per_coordinate_terms(state)):
        specs = iter_runtime_rotation_terms(
            getattr(term, "polynomial"),
            ignore_identity=bool(state.executor.ignore_identity),
            coefficient_tolerance=float(state.executor.coefficient_tolerance),
            sort_terms=bool(state.executor.sort_terms),
        )
        if len(specs) != 1:
            raise ValueError(
                "Resume requires one Pauli child per runtime coordinate; "
                f"coordinate {label!r} produced {len(specs)}."
            )
        spec = specs[0]
        coordinates.append(
            {
                "label": str(label),
                "pauli_exyz": str(spec.pauli_exyz),
                "coeff_real": float(spec.coeff_real),
                "nq": int(spec.nq),
            }
        )

    theta = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if theta.size != len(coordinates):
        raise ValueError(
            f"theta length {theta.size} does not match coordinate count {len(coordinates)}."
        )
    return {
        "schema": RESUME_SCHEMA_V1,
        "time": float(time),
        "seed_artifact_json": seed_artifact_json,
        "parameterization_mode": str(state.executor.parameterization_mode),
        "coordinates": coordinates,
        "theta_runtime": [float(v) for v in theta],
    }


def runtime_input_from_resume_state(
    resume: Mapping[str, Any],
    *,
    seed_artifact_json: str | None = None,
    base_runtime_input: Any | None = None,
) -> Any:
    """Rebuild a runtime input positioned at a previous run's final state.

    The problem definition and reference state are unchanged by propagation, so
    they come from the original seed: pass ``seed_artifact_json`` to load it, or
    ``base_runtime_input`` when it is already in hand (a caller continuing
    in-process, or a test that never wrote an artifact to disk).
    """

    from dataclasses import replace as dataclass_replace

    from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
    from pipelines.time_dynamics.ap_mclachlan.state import (
        _single_child_polynomial,
    )
    from src.quantum.ansatz_parameterization import build_parameter_layout
    from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm

    if str(resume.get("schema")) != RESUME_SCHEMA_V1:
        raise ValueError(f"Unsupported resume schema: {resume.get('schema')!r}")
    base = base_runtime_input
    if base is None:
        source = seed_artifact_json or resume.get("seed_artifact_json")
        if not source:
            raise ValueError(
                "Resume needs the originating seed artifact, or an explicit "
                "base_runtime_input, to supply the problem and reference state."
            )
        base = load_scaffold_runtime_input(source)
    terms = tuple(
        AnsatzTerm(
            label=str(c["label"]),
            polynomial=_single_child_polynomial(
                repr_mode="exyz",
                pauli_exyz=str(c["pauli_exyz"]),
                coeff_real=float(c["coeff_real"]),
                nq=int(c["nq"]),
            ),
            execution_mode="termwise_product",
        )
        for c in resume["coordinates"]
    )
    theta = np.asarray(resume["theta_runtime"], dtype=float).reshape(-1)
    layout = build_parameter_layout(terms)
    executor = CompiledAnsatzExecutor(
        terms,
        parameterization_layout=layout,
        parameterization_mode=str(resume.get("parameterization_mode", "per_pauli_term")),
    )
    psi_ref = np.asarray(base.psi_ref, dtype=complex).reshape(-1)
    psi_initial = np.asarray(executor.prepare_state(theta, psi_ref), dtype=complex)

    return dataclass_replace(
        base,
        psi_initial=psi_initial,
        base_layout=layout,
        theta_runtime=theta,
        theta_logical=np.zeros(int(layout.logical_parameter_count), dtype=float),
        selected_terms=terms,
        provenance={
            **dict(getattr(base, "provenance", {}) or {}),
            "resumed_from_time": float(resume["time"]),
            "resumed_coordinate_count": int(len(terms)),
        },
    )


__all__ = [
    "RESUME_SCHEMA_V1",
    "build_resume_state",
    "runtime_input_from_resume_state",
]
