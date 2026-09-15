"""Record and reload the conic problems the carried-witness guard solves.

The carried-witness radial moment flow hands Clarabel one conic problem per
solver-tier iteration: the adaptive spectral bundle tier, the literal full
62-row cone, the terminal completion polish, and the critical Schur tier.
This module stores each such problem with its exact operands -- the objective
matrix ``P``, the objective vector ``q``, the cone matrix ``A``, the cone
right-hand side ``b``, the cone specification, and the solver settings that
were set on the call -- together with the solution Clarabel returned and the
context needed to reconstruct the 62-row Gram of any candidate correction.

Nothing here runs unless a caller installs a recorder on the model, and the
recorder is off by default.  The stored instances are the input of the
offline replay bench ``cwrmf_solver_replay``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import clarabel
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ComplexArray = NDArray[np.complex128]

CONIC_INSTANCE_FORMAT = "cwrmf-conic-instance-v1"

TIER_NAMES = (
    "bundle_cone",
    "full_cone",
    "terminal_polish",
    "critical_schur",
)

#: The Clarabel settings fields the carried-witness call sites set explicitly.
#: Everything else is left at ``clarabel.DefaultSettings()``; the recorded
#: dictionary is what ``build_solver_settings`` replays.
RECORDED_SETTING_FIELDS = (
    "verbose",
    "max_threads",
    "max_iter",
    "tol_gap_abs",
    "tol_gap_rel",
    "tol_feas",
    "iterative_refinement_enable",
    "iterative_refinement_max_iter",
    "iterative_refinement_abstol",
    "iterative_refinement_reltol",
)

_CONE_CONSTRUCTORS = {
    "PSDTriangleConeT": clarabel.PSDTriangleConeT,
    "SecondOrderConeT": clarabel.SecondOrderConeT,
    "NonnegativeConeT": clarabel.NonnegativeConeT,
    "ZeroConeT": clarabel.ZeroConeT,
}


def solver_settings_dictionary(
    solver_settings: clarabel.DefaultSettings,
) -> dict[str, Any]:
    """Return the recorded subset of a Clarabel settings object."""

    return {
        name: getattr(solver_settings, name)
        for name in RECORDED_SETTING_FIELDS
    }


def build_solver_settings(
    recorded: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> clarabel.DefaultSettings:
    """Rebuild a Clarabel settings object from a record plus overrides.

    ``recorded`` restores the fields the call site set; ``overrides`` may set
    any attribute ``clarabel.DefaultSettings`` carries, so a candidate setting
    set is declared as a plain mapping.
    """

    settings = clarabel.DefaultSettings()
    for name, value in recorded.items():
        setattr(settings, name, value)
    if overrides:
        for name, value in overrides.items():
            if not hasattr(settings, name):
                raise ValueError(f"unknown Clarabel setting: {name}")
            setattr(settings, name, value)
    return settings


def cone_specification(cones: Sequence[Any]) -> tuple[tuple[str, int], ...]:
    """Return ``(constructor name, dimension)`` for each cone."""

    specification: list[tuple[str, int]] = []
    for cone in cones:
        name = type(cone).__name__
        if name.startswith("Py"):
            name = name[2:]
        if name not in _CONE_CONSTRUCTORS:
            raise ValueError(f"unrecorded cone type: {name}")
        specification.append((name, int(cone.dim)))
    return tuple(specification)


def build_cones(
    specification: Iterable[Sequence[Any]],
) -> list[Any]:
    """Rebuild the Clarabel cone list from a recorded specification."""

    return [
        _CONE_CONSTRUCTORS[str(name)](int(dimension))
        for name, dimension in specification
    ]


@dataclass(frozen=True)
class ConicInstance:
    """One conic problem handed to Clarabel, with its recorded solution.

    ``objective_matrix``/``objective_vector`` are ``P`` and ``q``;
    ``cone_matrix``/``cone_rhs`` are ``A`` and ``b``; ``cone_specification``
    names the cones in order.  ``base_velocity`` and ``correction_indices``
    map a returned ``x`` back to a full completion velocity in the 340
    completion coordinates: the candidate velocity is ``base_velocity`` with
    ``x`` added on ``correction_indices``.  ``gram_predictor`` is the shifted
    62-row unified Gram of the unconstrained endpoint, so the Gram of any
    candidate is ``gram_predictor`` plus the correction contracted with the
    step-scaled completion coefficient tensor stored once per run.
    """

    index: int
    tier: str
    atom_index: int
    atom_time: float
    iteration: int
    step_size: float
    objective_matrix: sparse.csc_matrix
    objective_vector: FloatArray
    cone_matrix: sparse.csc_matrix
    cone_rhs: FloatArray
    cone_specification: tuple[tuple[str, int], ...]
    solver_settings: dict[str, Any]
    base_velocity: FloatArray
    correction_indices: IntArray
    gram_predictor: ComplexArray
    desired_velocity: FloatArray
    completion_radius_squared: float
    retained_scaled_norm: float
    velocity_ceiling: float
    recorded_solution: FloatArray | None
    recorded_status: str
    recorded_objective: float
    recorded_primal_residual: float
    recorded_dual_residual: float
    recorded_iterations: int
    #: Clarabel's own reported ``solve_time`` for the recorded call, in
    #: seconds.  Replay wall time is measured by the bench, not read here.
    recorded_clarabel_solve_time: float

    @property
    def variable_count(self) -> int:
        return int(self.objective_vector.size)

    @property
    def constraint_count(self) -> int:
        return int(self.cone_rhs.size)

    def candidate_velocity(self, solution: FloatArray) -> FloatArray:
        """Return the full completion velocity implied by a returned ``x``."""

        candidate = np.array(self.base_velocity, dtype=float, copy=True)
        candidate[self.correction_indices] += np.asarray(
            solution, dtype=float
        )
        return candidate

    def reconstructed_gram(
        self,
        coefficients: ComplexArray,
        solution: FloatArray,
    ) -> ComplexArray:
        """Return the 62-row Gram of the candidate ``base + x``.

        ``coefficients`` is the step-scaled completion coefficient tensor of
        the run, shape ``(340, 62, 62)``.  The correction relative to the
        predictor is the candidate velocity minus the desired velocity, which
        is what the predictor Gram was built from.
        """

        correction = self.candidate_velocity(solution) - self.desired_velocity
        matrix = self.gram_predictor + np.einsum(
            "j,jab->ab", correction, coefficients, optimize=True
        )
        return 0.5 * (matrix + matrix.conjugate().T)


def _sparse_arrays(name: str, matrix: Any) -> dict[str, Any]:
    compressed = sparse.csc_matrix(matrix)
    return {
        f"{name}_data": compressed.data,
        f"{name}_indices": compressed.indices,
        f"{name}_indptr": compressed.indptr,
        f"{name}_shape": np.asarray(compressed.shape, dtype=np.int64),
    }


def _sparse_from(arrays: Any, name: str) -> sparse.csc_matrix:
    return sparse.csc_matrix(
        (
            arrays[f"{name}_data"],
            arrays[f"{name}_indices"],
            arrays[f"{name}_indptr"],
        ),
        shape=tuple(int(value) for value in arrays[f"{name}_shape"]),
    )


class ConicInstanceRecorder:
    """Write every guarded conic problem to a run directory.

    Each instance is written as it is solved, so the recorder holds no more
    than one problem in memory at a time.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, Any]] = []
        self._atom_index = -1
        self._atom_time = float("nan")
        self._write_manifest()

    @property
    def instance_count(self) -> int:
        return len(self._entries)

    @property
    def atom_index(self) -> int:
        return self._atom_index

    def begin_atom(self, time: float) -> None:
        """Open a new radial atom; instances are tagged with its index."""

        self._atom_index += 1
        self._atom_time = float(time)

    def write_context(
        self,
        *,
        step_scaled_completion_coefficients: ComplexArray,
        readable_completion_indices: IntArray,
        completion_count: int,
        numerical_profile: str,
        schur_safety_margin: float,
        solver_tolerance: float,
        psd_inflation: float,
    ) -> None:
        """Store the operands shared by every instance of the run."""

        np.savez(
            self.directory / "context.npz",
            step_scaled_completion_coefficients=np.asarray(
                step_scaled_completion_coefficients, dtype=complex
            ),
            readable_completion_indices=np.asarray(
                readable_completion_indices, dtype=np.int64
            ),
        )
        (self.directory / "context.json").write_text(
            json.dumps(
                {
                    "format": CONIC_INSTANCE_FORMAT,
                    "completion_count": int(completion_count),
                    "numerical_profile": str(numerical_profile),
                    "schur_safety_margin": float(schur_safety_margin),
                    "solver_tolerance": float(solver_tolerance),
                    "psd_inflation": float(psd_inflation),
                },
                indent=1,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def record(
        self,
        *,
        tier: str,
        iteration: int,
        step_size: float,
        objective_matrix: Any,
        objective_vector: Any,
        cone_matrix: Any,
        cone_rhs: Any,
        cones: Sequence[Any],
        solver_settings: clarabel.DefaultSettings,
        base_velocity: Any,
        correction_indices: Any,
        gram_predictor: Any,
        desired_velocity: Any,
        completion_radius_squared: float,
        retained_scaled_norm: float,
        velocity_ceiling: float,
        solution: Any,
        clarabel_solve_time: float,
    ) -> None:
        """Store one solved conic problem with its exact operands."""

        if tier not in TIER_NAMES:
            raise ValueError(f"unknown solver tier: {tier}")
        index = len(self._entries)
        arrays: dict[str, Any] = {
            "objective_vector": np.asarray(objective_vector, dtype=float),
            "cone_rhs": np.asarray(cone_rhs, dtype=float),
            "base_velocity": np.asarray(base_velocity, dtype=float),
            "correction_indices": np.asarray(
                correction_indices, dtype=np.int64
            ),
            "gram_predictor": np.asarray(gram_predictor, dtype=complex),
            "desired_velocity": np.asarray(desired_velocity, dtype=float),
        }
        arrays.update(_sparse_arrays("objective_matrix", objective_matrix))
        arrays.update(_sparse_arrays("cone_matrix", cone_matrix))
        returned = getattr(solution, "x", None)
        if returned is not None:
            arrays["recorded_solution"] = np.asarray(returned, dtype=float)
        np.savez(self.directory / f"instance_{index:04d}.npz", **arrays)
        entry = {
            "index": index,
            "tier": tier,
            "atom_index": self._atom_index,
            "atom_time": self._atom_time,
            "iteration": int(iteration),
            "step_size": float(step_size),
            "cone_specification": [
                [name, dimension]
                for name, dimension in cone_specification(cones)
            ],
            "solver_settings": solver_settings_dictionary(solver_settings),
            "completion_radius_squared": float(completion_radius_squared),
            "retained_scaled_norm": float(retained_scaled_norm),
            "velocity_ceiling": float(velocity_ceiling),
            "recorded_status": str(solution.status),
            "recorded_objective": float(solution.obj_val),
            "recorded_primal_residual": float(solution.r_prim),
            "recorded_dual_residual": float(solution.r_dual),
            "recorded_iterations": int(solution.iterations),
            "recorded_clarabel_solve_time": float(clarabel_solve_time),
            "has_solution": returned is not None,
        }
        self._entries.append(entry)
        self._write_manifest()

    def _write_manifest(self) -> None:
        temporary = self.directory / "instances.json.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "format": CONIC_INSTANCE_FORMAT,
                    "instances": self._entries,
                },
                indent=1,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.directory / "instances.json")


def load_instance_context(
    directory: Path | str,
) -> tuple[ComplexArray, IntArray, dict[str, Any]]:
    """Return the shared coefficient tensor, readable indices and metadata."""

    path = Path(directory)
    arrays = np.load(path / "context.npz")
    metadata = json.loads((path / "context.json").read_text(encoding="utf-8"))
    return (
        np.asarray(
            arrays["step_scaled_completion_coefficients"], dtype=complex
        ),
        np.asarray(arrays["readable_completion_indices"], dtype=np.int64),
        metadata,
    )


def load_instances(directory: Path | str) -> list[ConicInstance]:
    """Return every recorded conic instance in solve order."""

    path = Path(directory)
    manifest = json.loads(
        (path / "instances.json").read_text(encoding="utf-8")
    )
    if manifest.get("format") != CONIC_INSTANCE_FORMAT:
        raise ValueError("unknown conic instance format")
    instances: list[ConicInstance] = []
    for entry in manifest["instances"]:
        index = int(entry["index"])
        arrays = np.load(path / f"instance_{index:04d}.npz")
        solution = (
            np.asarray(arrays["recorded_solution"], dtype=float)
            if entry["has_solution"]
            else None
        )
        instances.append(
            ConicInstance(
                index=index,
                tier=str(entry["tier"]),
                atom_index=int(entry["atom_index"]),
                atom_time=float(entry["atom_time"]),
                iteration=int(entry["iteration"]),
                step_size=float(entry["step_size"]),
                objective_matrix=_sparse_from(arrays, "objective_matrix"),
                objective_vector=np.asarray(
                    arrays["objective_vector"], dtype=float
                ),
                cone_matrix=_sparse_from(arrays, "cone_matrix"),
                cone_rhs=np.asarray(arrays["cone_rhs"], dtype=float),
                cone_specification=tuple(
                    (str(name), int(dimension))
                    for name, dimension in entry["cone_specification"]
                ),
                solver_settings=dict(entry["solver_settings"]),
                base_velocity=np.asarray(
                    arrays["base_velocity"], dtype=float
                ),
                correction_indices=np.asarray(
                    arrays["correction_indices"], dtype=np.int64
                ),
                gram_predictor=np.asarray(
                    arrays["gram_predictor"], dtype=complex
                ),
                desired_velocity=np.asarray(
                    arrays["desired_velocity"], dtype=float
                ),
                completion_radius_squared=float(
                    entry["completion_radius_squared"]
                ),
                retained_scaled_norm=float(entry["retained_scaled_norm"]),
                velocity_ceiling=float(entry["velocity_ceiling"]),
                recorded_solution=solution,
                recorded_status=str(entry["recorded_status"]),
                recorded_objective=float(entry["recorded_objective"]),
                recorded_primal_residual=float(
                    entry["recorded_primal_residual"]
                ),
                recorded_dual_residual=float(entry["recorded_dual_residual"]),
                recorded_iterations=int(entry["recorded_iterations"]),
                recorded_clarabel_solve_time=float(
                    entry["recorded_clarabel_solve_time"]
                ),
            )
        )
    return instances
