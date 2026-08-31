"""Continuation of a completed trajectory from its final state.

Long horizons are reached by continuing, not by re-running: the expensive cells
of a production matrix take hours to reach their end time, and extending them
from t=0 discards that work. A continuation needs the propagated state, not the
original seed, so a completed run records everything required to rebuild its
final support exactly.

What must be preserved is the implemented unitary.  In per-Pauli mode that is
the ordered list of single-Pauli runtime coordinates.  In logical-shared mode
one coordinate is an intact macro generator, so its full ordered rotation list,
label, execution mode, and shared angle must travel together.  The reference
state is unchanged by propagation, so it comes from the original seed artifact
rather than being duplicated here.

Controller state is recorded separately from the physical ansatz state.  The
physical reconstruction helper remains useful to cost diagnostics and other
read-only consumers, while the trajectory adapter also restores deletion
history, cooldowns, solve-repair hysteresis, the prior accepted velocity, and
the global checkpoint index.  A promoted t=5 -> t=20 run can therefore make
the same decisions as an uninterrupted run on the same global grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

RESUME_SCHEMA_V1 = "paper_ii_resume_state_v1"
RESUME_SCHEMA_V2 = "paper_ii_resume_state_v2"
CONTROLLER_RESUME_SCHEMA_V1 = "paper_ii_controller_resume_state_v1"


@dataclass(frozen=True)
class ControllerResumeState:
    """JSON-safe decision state immediately before a trajectory checkpoint."""

    checkpoint_index: int
    prune_runtime_state: Mapping[str, Any]
    repair_hold_state: Mapping[str, Any] | None
    previous_accepted_theta_dot: tuple[float, ...] | None

    def __post_init__(self) -> None:
        if int(self.checkpoint_index) < 0:
            raise ValueError("checkpoint_index must be nonnegative")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTROLLER_RESUME_SCHEMA_V1,
            "checkpoint_index": int(self.checkpoint_index),
            "prune_runtime_state": dict(self.prune_runtime_state),
            "repair_hold_state": (
                None
                if self.repair_hold_state is None
                else dict(self.repair_hold_state)
            ),
            "previous_accepted_theta_dot": (
                None
                if self.previous_accepted_theta_dot is None
                else [float(value) for value in self.previous_accepted_theta_dot]
            ),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "ControllerResumeState":
        if str(payload.get("schema")) != CONTROLLER_RESUME_SCHEMA_V1:
            raise ValueError(
                f"Unsupported controller resume schema: {payload.get('schema')!r}"
            )
        previous = payload.get("previous_accepted_theta_dot")
        return cls(
            checkpoint_index=int(payload["checkpoint_index"]),
            prune_runtime_state=dict(payload.get("prune_runtime_state") or {}),
            repair_hold_state=(
                None
                if payload.get("repair_hold_state") is None
                else dict(payload["repair_hold_state"])
            ),
            previous_accepted_theta_dot=(
                None
                if previous is None
                else tuple(float(value) for value in previous)
            ),
        )


def build_resume_state(
    state: Any,
    *,
    theta_runtime: Sequence[float],
    time: float,
    seed_artifact_json: str | None,
    controller_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize the end of a trajectory so it can be continued exactly."""

    from pipelines.time_dynamics.ap_mclachlan.state import _per_coordinate_terms
    from src.quantum.ansatz_parameterization import iter_runtime_rotation_terms

    mode = str(
        getattr(
            state,
            "parameterization_mode",
            getattr(state.executor, "parameterization_mode", "per_pauli_term"),
        )
    )
    problem_family = str(
        getattr(
            getattr(state, "resolved_problem", None),
            "family_key",
            "",
        )
    ).strip().lower()
    coordinate_terms = tuple(_per_coordinate_terms(state))
    theta = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if theta.size != len(coordinate_terms):
        raise ValueError(
            f"theta length {theta.size} does not match coordinate count "
            f"{len(coordinate_terms)}."
        )

    if mode == "logical_shared":
        coordinates: list[dict[str, Any]] = []
        runtime_labels = tuple(state.runtime_coordinate_labels)
        if len(runtime_labels) != len(coordinate_terms):
            raise ValueError(
                "Logical resume label count does not match coordinate count: "
                f"{len(runtime_labels)} vs {len(coordinate_terms)}."
            )
        for runtime_label, term in zip(runtime_labels, coordinate_terms):
            specs = iter_runtime_rotation_terms(
                getattr(term, "polynomial"),
                ignore_identity=bool(state.executor.ignore_identity),
                coefficient_tolerance=float(state.executor.coefficient_tolerance),
                sort_terms=bool(state.executor.sort_terms),
            )
            coordinates.append(
                {
                    "label": str(getattr(term, "label", runtime_label)),
                    "runtime_label": str(runtime_label),
                    "execution_mode": str(
                        getattr(term, "execution_mode", "termwise_product")
                        or "termwise_product"
                    ),
                    "rotations": [
                        {
                            "pauli_exyz": str(spec.pauli_exyz),
                            "coeff_real": float(spec.coeff_real),
                            "nq": int(spec.nq),
                        }
                        for spec in specs
                    ],
                }
            )
        payload = {
            "schema": RESUME_SCHEMA_V2,
            "time": float(time),
            "seed_artifact_json": seed_artifact_json,
            "problem_family": problem_family or None,
            "parameterization_mode": mode,
            "layout": {
                "term_order": str(state.layout.term_order),
                "ignore_identity": bool(state.layout.ignore_identity),
                "coefficient_tolerance": float(
                    state.layout.coefficient_tolerance
                ),
            },
            "coordinates": coordinates,
            "theta_runtime": [float(value) for value in theta],
            "theta_logical": [float(value) for value in theta],
        }
        if controller_state is not None:
            controller = ControllerResumeState.from_json_dict(controller_state)
            payload["controller_state"] = controller.to_json_dict()
        return payload

    coordinates = []
    for label, term in zip(state.runtime_coordinate_labels, coordinate_terms):
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

    payload = {
        "schema": RESUME_SCHEMA_V1,
        "time": float(time),
        "seed_artifact_json": seed_artifact_json,
        "parameterization_mode": mode,
        "coordinates": coordinates,
        "theta_runtime": [float(v) for v in theta],
    }
    if controller_state is not None:
        controller = ControllerResumeState.from_json_dict(controller_state)
        payload["controller_state"] = controller.to_json_dict()
    return payload


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
    from pipelines.time_dynamics.ap_mclachlan.state import _single_child_polynomial
    from src.quantum.ansatz_parameterization import (
        build_parameter_layout,
        expand_legacy_logical_theta,
    )
    from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.qubitization_module import PauliTerm
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm

    schema = str(resume.get("schema"))
    if schema not in {RESUME_SCHEMA_V1, RESUME_SCHEMA_V2}:
        raise ValueError(f"Unsupported resume schema: {resume.get('schema')!r}")
    declared_family = str(resume.get("problem_family") or "").strip().lower()
    base = base_runtime_input
    if base is None:
        source = seed_artifact_json or resume.get("seed_artifact_json")
        if not source:
            raise ValueError(
                "Resume needs the originating seed artifact, or an explicit "
                "base_runtime_input, to supply the problem and reference state."
            )
        if schema == RESUME_SCHEMA_V2 and declared_family == "siam":
            from pipelines.time_dynamics.siam_seed import load_siam_runtime_input

            base = load_siam_runtime_input(source)
        else:
            base = load_scaffold_runtime_input(source)
    base_family = str(
        getattr(
            getattr(base, "resolved_problem", None),
            "family_key",
            "",
        )
    ).strip().lower()
    if declared_family and base_family != declared_family:
        raise ValueError(
            "Resume problem family does not match the originating seed: "
            f"{declared_family!r} vs {base_family!r}."
        )
    mode = str(resume.get("parameterization_mode", "per_pauli_term"))
    coordinates = resume.get("coordinates")
    if not isinstance(coordinates, Sequence) or isinstance(
        coordinates, (str, bytes)
    ):
        raise ValueError("Resume coordinates must be a sequence.")

    if schema == RESUME_SCHEMA_V1:
        terms = tuple(
            AnsatzTerm(
                label=str(coordinate["label"]),
                polynomial=_single_child_polynomial(
                    repr_mode="exyz",
                    pauli_exyz=str(coordinate["pauli_exyz"]),
                    coeff_real=float(coordinate["coeff_real"]),
                    nq=int(coordinate["nq"]),
                ),
                execution_mode="termwise_product",
            )
            for coordinate in coordinates
        )
        layout = build_parameter_layout(terms)
        theta_runtime = np.asarray(
            resume["theta_runtime"], dtype=float
        ).reshape(-1)
        theta_logical = (
            theta_runtime.copy()
            if mode == "logical_shared"
            else np.zeros(int(layout.logical_parameter_count), dtype=float)
        )
    else:
        if mode != "logical_shared":
            raise ValueError(
                f"Resume schema {RESUME_SCHEMA_V2!r} requires "
                "parameterization_mode='logical_shared'."
            )
        terms_out: list[AnsatzTerm] = []
        for coordinate in coordinates:
            rotations = coordinate.get("rotations")
            if not isinstance(rotations, Sequence) or isinstance(
                rotations, (str, bytes)
            ):
                raise ValueError(
                    f"Logical resume coordinate {coordinate.get('label')!r} "
                    "is missing its ordered rotation list."
                )
            polynomial = PauliPolynomial("exyz")
            for rotation in rotations:
                polynomial.add_term(
                    PauliTerm(
                        int(rotation["nq"]),
                        ps=str(rotation["pauli_exyz"]),
                        pc=float(rotation["coeff_real"]),
                    )
                )
            terms_out.append(
                AnsatzTerm(
                    label=str(coordinate["label"]),
                    polynomial=polynomial,
                    execution_mode=str(
                        coordinate.get("execution_mode", "termwise_product")
                        or "termwise_product"
                    ),
                )
            )
        terms = tuple(terms_out)
        layout_contract = dict(resume.get("layout") or {})
        term_order = str(layout_contract.get("term_order", "sorted"))
        if term_order not in {"sorted", "native"}:
            raise ValueError(
                f"Unsupported logical resume term_order: {term_order!r}."
            )
        layout = build_parameter_layout(
            terms,
            ignore_identity=bool(layout_contract.get("ignore_identity", True)),
            coefficient_tolerance=float(
                layout_contract.get("coefficient_tolerance", 1.0e-12)
            ),
            sort_terms=(term_order == "sorted"),
        )
        theta_runtime = np.asarray(
            resume["theta_runtime"], dtype=float
        ).reshape(-1)
        theta_logical = np.asarray(
            resume["theta_logical"], dtype=float
        ).reshape(-1)
        expected = int(layout.logical_parameter_count)
        if theta_runtime.size != expected or theta_logical.size != expected:
            raise ValueError(
                "Logical resume angle count does not match macro count: "
                f"theta_runtime={theta_runtime.size}, "
                f"theta_logical={theta_logical.size}, expected={expected}."
            )
        if not np.array_equal(theta_runtime, theta_logical):
            raise ValueError(
                "Logical resume theta_runtime and theta_logical must agree "
                "coordinate by coordinate."
            )
        from pipelines.time_dynamics.ap_mclachlan.state import (
            runtime_coordinate_labels,
        )

        rebuilt_labels = runtime_coordinate_labels(
            layout,
            parameterization_mode=mode,
        )
        declared_labels = tuple(
            str(coordinate.get("runtime_label", rebuilt))
            for coordinate, rebuilt in zip(coordinates, rebuilt_labels)
        )
        if declared_labels != rebuilt_labels:
            raise ValueError(
                "Logical resume runtime labels do not match reconstructed "
                f"macro labels: {declared_labels!r} vs {rebuilt_labels!r}."
            )

    parameter_theta = (
        theta_logical if mode == "logical_shared" else theta_runtime
    )
    expected_parameter_count = (
        int(layout.logical_parameter_count)
        if mode == "logical_shared"
        else int(layout.runtime_parameter_count)
    )
    if parameter_theta.size != expected_parameter_count:
        raise ValueError(
            f"Resume theta length {parameter_theta.size} does not match "
            f"parameter count {expected_parameter_count} for mode {mode!r}."
        )
    scaffold_theta_runtime = (
        expand_legacy_logical_theta(theta_logical, layout)
        if mode == "logical_shared"
        else theta_runtime
    )
    executor = CompiledAnsatzExecutor(
        terms,
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_layout=layout,
        parameterization_mode=mode,
    )
    psi_ref = np.asarray(base.psi_ref, dtype=complex).reshape(-1)
    psi_initial = np.asarray(
        executor.prepare_state(parameter_theta, psi_ref), dtype=complex
    )

    return dataclass_replace(
        base,
        psi_initial=psi_initial,
        base_layout=layout,
        theta_runtime=scaffold_theta_runtime,
        theta_logical=theta_logical,
        selected_terms=terms,
        provenance={
            **dict(getattr(base, "provenance", {}) or {}),
            "resumed_from_time": float(resume["time"]),
            "resumed_coordinate_count": int(len(terms)),
            "resume_schema": schema,
            "resume_parameterization_mode": mode,
        },
    )


__all__ = [
    "CONTROLLER_RESUME_SCHEMA_V1",
    "ControllerResumeState",
    "RESUME_SCHEMA_V1",
    "RESUME_SCHEMA_V2",
    "build_resume_state",
    "runtime_input_from_resume_state",
]
